import torch
import torch.nn.functional as F
import itertools
import warnings
import logging
from copy import deepcopy

from utils_prediction.nn.models import (
    TorchModel,
    FixedWidthModel,
    BilevelModel,
)

from utils_prediction.nn.layers import FeedforwardNet
from utils_prediction.nn.pytorch_metrics import (
    sigmoid,
    weighted_mean,
    roc_auc_score_surrogate,
    precision_surrogate,
    tpr_surrogate,
    fpr_surrogate,
    positive_rate_surrogate,
    
    # average_precision_surrogate,
    weighted_cross_entropy_loss,
    MetricUndefinedError,
    baselined_loss,
)

# Structure:
# group_regularized_model -> function returning a model class based on a provided key
# GroupRegularizedModel -> upper level model class. Defines a regularized loss that can be computed arbitrarily
#   GroupCoralModel -> penalizes difference in mean & covariance of layer activation across groups
#   MMDModel -> penalizes MMD between model predictions for conditional prediction parity
#   EqualMeanPredictionModel -> penalizes mean difference in predictions with same interface as MMDModel
#   GroupIRMModel -> Applies the IRM penalty to each group
#   EqualThresholdRateModel -> penalizes thresholded prediction rates across groups with same interface as MMDModel
#   GroupCalibrationModel -> penalizes difference in calibration curve across groups with surrogate model
# GroupMetricRegularizedModel -> penalizes the differences in a differentiable metric across groups
#   EqualAUCModel
#   EqualPrecisionModel
#   EqualAPModel
#   EqualLossModel
#   EqualBrierScoreModel
#   EqualTPRModel
#   EqualFPRModel
#   EqualPositiveRateModel
# GroupAdversarialModel -> penalizes difference in distribution of predictions (or layer activation) across groups w/ discriminator


def group_regularized_model(model_type="loss"):
    """
    A function that returns an instance of GroupRegularizedModel
    """
    class_dict = {
        "adversarial": GroupAdversarialModel,
        "group_irm": GroupIRMModel,
        "group_coral": GroupCoralModel,
    }
    the_class = class_dict.get(model_type, None)
    if the_class is None:
        raise ValueError("model_type not defined in group_regularized_model")
    return the_class


class GroupRegularizedModel(TorchModel):
    """
    A model that penalizes differences in a quantity across groups
    """

    def compute_group_regularization_loss(
        self, outputs, labels, group, sample_weight=None
    ):
        """
        Computes a regularization term defined in terms of model outputs, labels, and group.
        This class should be overriden and this regularization term defined.
        """
        raise NotImplementedError

    def get_default_config(self):
        """
        Default parameters
        """
        config_dict = super().get_default_config()
        update_dict = {
            "num_hidden": 1,
            "hidden_dim": 128,
            "drop_prob": 0.0,
            "normalize": False,
            "sparse": True,
            "sparse_mode": "csr",  # alternatively, "convert"
            "resnet": False,
        }
        return {**config_dict, **update_dict}

    def init_model(self):
        model = FeedforwardNet(
            in_features=self.config_dict["input_dim"],
            hidden_dim_list=self.config_dict["num_hidden"]
            * [self.config_dict["hidden_dim"]],
            output_dim=self.config_dict["output_dim"],
            drop_prob=self.config_dict["drop_prob"],
            normalize=self.config_dict["normalize"],
            sparse=self.config_dict["sparse"],
            sparse_mode=self.config_dict["sparse_mode"],
            resnet=self.config_dict["resnet"],
        )
        return model

    def get_transform_batch_keys(self):
        """
        Returns the names of the list of tensors that sent to device
        """
        result = super().get_transform_batch_keys()
        result = result + ["group"]

    def get_loss_names(self):
        return ["loss", "supervised", "group_regularization"]

    def forward_on_batch(self, the_data):
        """
        Run the forward pass, returning a batch_loss_dict and outputs
        """
        loss_dict_batch = {}
        inputs, labels, group = (
            the_data["features"],
            the_data["labels"],
            the_data["group"],
        )
        outputs = self.model(inputs)
        
        # Compute the loss
        if self.config_dict.get("weighted_loss"):
            if self.config_dict.get("domain_adapt"):
                #### hack to make domain adaptation work within domain gen 
                ## Classifier weights
                # target domain(group 1): 0
                # source domain(group 0): batch_size/number_of_source_domain_samples
                ## Discriminator weights
                # all 1
                sample_weight = torch.zeros(group.size()).to(self.device)
                sample_weight[group==0] = len(group)/torch.sum(group==0)
                
                loss_dict_batch = {
                    "supervised": self.criterion(
                        outputs, labels, sample_weight=sample_weight
                    ),
                    "group_regularization": self.compute_group_regularization_loss(
                        outputs,
                        labels,
                        group
                    ),
                }
            else:
                loss_dict_batch["supervised"] = self.criterion(
                    outputs, labels, sample_weight=the_data.get("weights")
                )
                loss_dict_batch[
                    "group_regularization"
                ] = self.compute_group_regularization_loss(
                    outputs, labels, group, sample_weight=the_data.get("weights")
                )
        else:
            loss_dict_batch["supervised"] = self.criterion(outputs, labels)
            loss_dict_batch[
                "group_regularization"
            ] = self.compute_group_regularization_loss(outputs, labels, group)

        loss_dict_batch["loss"] = loss_dict_batch["supervised"] + (
            self.config_dict["lambda_group_regularization"]
            * loss_dict_batch["group_regularization"]
        )
        return loss_dict_batch, outputs

class GroupCoralModel(GroupRegularizedModel):
    """
    Aligns means and covariances of features from different groups. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # register forward hook
        self.init_hook(layer = self.config_dict['which_layer'])
    
    def get_default_config(self):
        config_dict = super().get_default_config()
        update_dict = {
            "which_layer": -1, # last hidden layer
            "lambda_group_regularization": 1e-1,
            "group_regularization_mode": "group",
            "domain_adapt":False
        }
        return {**config_dict, **update_dict}
    
    def init_hook(self, layer = -1):
        """
        register forward hook to LinearLayerWrapper
        """
        def get_activations(mod, inp, out):
            self.activations=out
            
        if layer == 0: layer = -self.config_dict['num_hidden']
        
        which_layer = self.config_dict['num_hidden'] + layer
        for name,l in self.model.named_modules():
            if f"layers.{which_layer}" in name and "LinearLayerWrapper" in str(type(l)):
                l.register_forward_hook(get_activations)
    
    def compute_penalty(self, x, y):
        if x.dim() > 2:
            # if layer output Tensors of size (batch_size, ..., feature dimensionality).
            # flatten to Tensors of size (*, feature dimensionality)
            x = x.view(-1, x.size(-1))
            y = y.view(-1, y.size(-1))
        
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)
        
        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()
        #print(mean_diff, cova_diff)
        
        penalty = mean_diff+cova_diff
        
        # if group size == 1 then output can result in NaN
        if torch.isnan(penalty): penalty = 0 
        return penalty
    
    def compute_penalty_group(self, group, sample_weight):
        unique_groups = group.unique()
        penalty = torch.tensor([0.0], dtype=torch.float).to(self.device)
        if len(unique_groups) == 1:
            return penalty
        i = 0
        if self.config_dict["group_regularization_mode"] == "overall":
            for the_group in unique_groups:
                penalty = penalty + self.compute_penalty(
                    self.activations[group == the_group],
                    self.activations[group != the_group]
                )
                i = i + 1
        elif self.config_dict["group_regularization_mode"] == "group":
            for comb in itertools.combinations(unique_groups, 2):
                penalty = penalty + self.compute_penalty(
                    self.activations[group == comb[0]],
                    self.activations[group == comb[1]]
                )
                i = i + 1
        return penalty / i
    
    def compute_group_regularization_loss(
        self, outputs, labels, group, sample_weight=None
    ):
        """
        Partition the data on the labels and compute the penalty
        (TODO) Implement sample_weight
        """
        penalty = torch.FloatTensor([0.0]).to(self.device)
        penalty = penalty + self.compute_penalty_group(
            group,
            sample_weight=sample_weight if sample_weight is not None else None,
        )
        return penalty


class GroupIRMModel(GroupRegularizedModel):
    """
    Applies IRMv1 over groups
    """

    def compute_penalty(self, outputs, labels):
        # https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/colored_mnist/main.py#L107

        scale = torch.FloatTensor([1.0]).requires_grad_().to(self.device)
        loss = self.criterion(outputs * scale, labels)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def compute_group_regularization_loss(
        self, outputs, labels, group, sample_weight=None
    ):
        # (TODO) implemented sample_weight
        result = torch.FloatTensor([0.0]).to(self.device)
        unique_groups = group.unique()
        for i, the_group in enumerate(unique_groups):
            the_penalty = self.compute_penalty(
                outputs[group == the_group], labels[group == the_group]
            )
            result = result + the_penalty
            logging.debug(f"Group: {the_group}, Penalty: {the_penalty}")
        return result




## Bilevel optimization models ##
class GroupAdversarialModel(BilevelModel, FixedWidthModel):
    def __init__(self, *args, **kwargs):
        if kwargs.get("output_dim_discriminator") is None:
            kwargs["output_dim_discriminator"] = kwargs.get("num_groups")
        print(kwargs["output_dim_discriminator"])
        super().__init__(*args, **kwargs)
        
        # register forward hook
        if self.config_dict['use_layer_activations'] and self.config_dict['num_hidden']>0:
            self.init_hook(layer = self.config_dict['which_layer'])

    def get_default_config(self):
        """
        Defines default hyperparameters that may be overwritten.
        """
        config_dict = super().get_default_config()
        update_dict = {
            "adversarial_mode": "unconditional",
            "lr_discriminator": 1e-3,
            "lambda_group_regularization": 1e-1,
            "num_hidden_discriminator": 1,
            "hidden_dim_discriminator": 32,
            "output_dim_discriminator": None,  # specify at initialization
            "drop_prob_discriminator": 0.0,
            "normalize_discriminator": False,
            "spectral_norm": True,
            "sparse": True,
            "sparse_mode": "csr",
            "print_grads": False,
            "use_layer_activations": False, # layer activation as input to discriminator
            "which_layer": -1, # take activations from last hidden layer by default
            "shuffle_group_labels": False, # if True, objective:= min +loss(discrim) instead of min -loss(discrim)
            "reverse_gradients": False, # if True, objective:= min +loss(discrim) instead of min -loss(discrim)
            "reverse_group_labels": False, # for domain adaptation (assumes 2 groups) 
                                           # if True, objective:= min +loss(discrim) instead of min -loss(discrim)
        }

        return {**config_dict, **update_dict}

    def print_grads(self):
        for name, param in self.models_aux["discriminator"].named_parameters():
            if param.requires_grad and param.grad is not None:
                logging.info("{}: {}".format(name, param.grad.mean()))

    def get_loss_names(self):
        return ["loss", "supervised", "discriminator", "discriminator_alt"]
    
    def init_hook(self, layer = -1):
        """
        register forward hook to LinearLayerWrapper
        """
        def get_activations(mod, inp, out):
            self.activations=out
        
        if layer == 0: layer = -self.config_dict['num_hidden']
        
        which_layer = self.config_dict['num_hidden'] + layer
        for name,l in self.model.named_modules():
            if f"layers.{which_layer}" in name and "LinearLayerWrapper" in str(type(l)):
                l.register_forward_hook(get_activations)

    def init_optimizers_aux(self):
        return {
            "discriminator": torch.optim.Adam(
                [{"params": self.models_aux["discriminator"].parameters()}],
                lr=self.config_dict["lr_discriminator"],
            )
        }

    def init_models_aux(self):
        
        if self.config_dict['use_layer_activations']:
            in_dim = self.config_dict['hidden_dim']
        else:
            in_dim = self.config_dict['output_dim']
        
        models_aux = {
            "discriminator": FeedforwardNet(
                in_features=in_dim
                + (1 * (self.config_dict["adversarial_mode"] == "conditional")),
                hidden_dim_list=self.config_dict["num_hidden_discriminator"]
                * [self.config_dict["hidden_dim_discriminator"]],
                output_dim=self.config_dict["output_dim_discriminator"],
                drop_prob=self.config_dict["drop_prob_discriminator"],
                normalize=self.config_dict["normalize_discriminator"],
                sparse=False,
                sparse_mode=None,
                resnet=self.config_dict.get("resnet", False),
                spectral_norm=self.config_dict.get("spectral_norm", False),
                add_revgrad_to_start=self.config_dict["reverse_gradients"],
                revgrad_lambd=self.config_dict["lambda_group_regularization"]
            )
        }
        for model in models_aux.values():
            model.apply(self.weights_init)
            model.to(self.device)
        return models_aux

    def get_transform_batch_keys(self):
        """
        Returns the names of the list of tensors that sent to device
        """
        result = super().get_transform_batch_keys()
        result = result + ["group"]

    def forward_on_batch_helper(self, the_data):
        # Run data through the model
        outputs = self.model(the_data["features"])
        
        # Run data through the discriminator
        if self.config_dict["adversarial_mode"] == "conditional":
            if self.config_dict["use_layer_activations"]:
                inputs_discriminator = torch.cat(
                    (
                        self.activations,
                        torch.unsqueeze(the_data["labels"].to(torch.float), dim=1),
                    ),
                    dim=1,
                )
            else:
                inputs_discriminator = torch.cat(
                    (
                        F.log_softmax(outputs, dim=1),
                        torch.unsqueeze(the_data["labels"].to(torch.float), dim=1),
                    ),
                    dim=1,
                )
        else:
            if self.config_dict["use_layer_activations"]:
                inputs_discriminator = self.activations
            else:
                inputs_discriminator = F.log_softmax(outputs, dim=1)

        outputs_discriminator = self.models_aux["discriminator"](inputs_discriminator)
        
        if self.config_dict.get("weighted_loss"):
            if self.config_dict.get("domain_adapt"):
                #### hack to make domain adaptation work within domain gen 
                ## Classifier weights
                # target domain(group 1): 0
                # source domain(group 0): batch_size/number_of_source_domain_samples
                ## Discriminator weights
                # all 1
                sample_weight = torch.zeros(the_data['group'].size()).to(self.device)
                sample_weight[the_data['group']==0] = len(the_data['group'])/torch.sum(the_data['group']==0)
                
                loss_dict_batch = {}
                # classifier loss - ignore group 1
                loss_dict_batch['supervised'] = self.criterion(
                    outputs, the_data["labels"], sample_weight=sample_weight
                )
                
                # shuffle or reverse weight for discriminator if specified
                if self.config_dict["shuffle_group_labels"]:
                    if self.is_training():
                        the_data['group'] = the_data['group'][
                            torch.randperm(the_data['group'].size()[0])
                        ]
        
                if self.config_dict['reverse_group_labels']:
                    if self.is_training():
                        the_data['group'] = the_data['group']+1
                        the_data['group'][the_data['group']==2]=0
                
                # discriminator loss
                loss_dict_batch['discriminator'] = self.criterion(
                    outputs_discriminator,
                    the_data["group"]
                )
                
            else:
                loss_dict_batch = {
                    "supervised": self.criterion(
                        outputs, the_data["labels"], sample_weight=the_data["weights"]
                    ),
                    "discriminator": self.criterion(
                        outputs_discriminator,
                        the_data["group"],
                        sample_weight=the_data["weights"],
                    ),
                }
        else:
            loss_dict_batch = {
                "supervised": self.criterion(outputs, the_data["labels"]),
                "discriminator": self.criterion(
                    outputs_discriminator, the_data["group"]
                ),
            }
        loss_dict_batch["discriminator_alt"] = torch.exp(
            -loss_dict_batch["discriminator"]
        )
        return loss_dict_batch, outputs

    def forward_on_batch(self, the_data):
        loss_dict_batch, outputs = self.forward_on_batch_helper(the_data)
        
        if self.config_dict["shuffle_group_labels"] or self.config_dict['reverse_group_labels']: 
            loss_dict_batch["loss"] = (
                loss_dict_batch["supervised"]
                + self.config_dict["lambda_group_regularization"]
                * loss_dict_batch["discriminator"]
            )
        elif self.config_dict['reverse_gradients']:
            loss_dict_batch["loss"] = (
                loss_dict_batch["supervised"]
                + loss_dict_batch["discriminator"]
            )
        else:
            loss_dict_batch["loss"] = (
                loss_dict_batch["supervised"]
                - self.config_dict["lambda_group_regularization"]
                * loss_dict_batch["discriminator"]
            )

        return loss_dict_batch, outputs

    def update_models_aux(self, the_data):
        loss_dict_batch, _ = self.forward_on_batch_helper(the_data)
        loss_dict_batch["discriminator"].backward()
        self.optimizers_aux["discriminator"].step()


class GroupCalibrationInvarianceModel(BilevelModel, GroupRegularizedModel):
    def get_default_config(self):
        """
        Defines default hyperparameters that may be overwritten.
        """
        config_dict = super().get_default_config()

        update_dict = {"lr_aux": 1e-4}
        return {**config_dict, **update_dict}

    def get_model_aux(self):
        # return LinearLayerWrapper(in_features=1, out_features=2)

        return FeedforwardNet(
            in_features=1,
            hidden_dim_list=[16],
            output_dim=2,
            drop_prob=0.0,
            normalize=0.0,
            sparse=False,
        )

    def init_models_aux(self):
        models_aux = {
            i: self.get_model_aux()
            for i in range(
                -1, self.config_dict["num_groups"]
            )  # -1 refers to the marginal model
        }

        for model in models_aux.values():
            model.apply(self.weights_init)
            model.to(self.device)
        return models_aux

    def init_optimizers_aux(self):
        return {
            i: torch.optim.Adam(
                [{"params": self.models_aux[i].parameters()}],
                lr=self.config_dict["lr_aux"],
            )
            for i in range(
                -1, self.config_dict["num_groups"]
            )  # -1 refers to the marginal model
        }

    def compute_group_regularization_loss(
        self, outputs, labels, group, sample_weight=None
    ):
        # (TODO) implement sample_weight
        result = torch.FloatTensor().to(self.device)
        outputs = F.log_softmax(outputs, dim=1)[:, -1].unsqueeze(1)

        for the_group in group.unique():
            outputs_group = outputs[group == the_group]
            log_probs_group = F.softmax(
                self.models_aux[int(the_group)](outputs_group), dim=1
            )[:, 1]
            log_probs_marginal = F.softmax(self.models_aux[-1](outputs_group), dim=1)[
                :, 1
            ]
            result = torch.cat((result, log_probs_group - log_probs_marginal))
        result = torch.square(result).mean()

        return result

    def update_models_aux(self, the_data):

        outputs = self.model(the_data["features"])
        outputs = F.log_softmax(outputs, dim=1)[:, -1].unsqueeze(1)

        outputs_marginal = self.models_aux[-1](outputs)

        loss_marginal = self.criterion(outputs_marginal, the_data["labels"])

        loss_marginal.backward(retain_graph=True)
        self.optimizers_aux[-1].step()

        for the_group in the_data["group"].unique():
            outputs_group = self.models_aux[int(the_group)](
                outputs[the_data["group"] == the_group]
            )
            loss_group = self.criterion(
                outputs_group, the_data["labels"][the_data["group"] == the_group]
            )
            loss_group.backward(retain_graph=True)
            self.optimizers_aux[int(the_group)].step()
