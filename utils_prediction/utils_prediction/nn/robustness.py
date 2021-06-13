import numpy as np
import torch
import logging

from utils_prediction.nn.models import (
    TorchModel,
    FeedforwardNet,
)
from utils_prediction.nn.pytorch_metrics import (
    MetricUndefinedError,
    weighted_cross_entropy_loss,
    indicator,
    logistic_surrogate,
    roc_auc_score_surrogate,
    IRM_penalty,
    baselined_loss,
)


def group_robust_model(model_type="loss"):
    """
    A function that returns an instance of GroupRegularizedModel
    """
    class_dict = {
        "loss": GroupDROModel,
        "baselined_loss": GroupDROBaselinedLoss,
        "auc": GroupDROAUC,
        "auc_proxy": GroupDROAUCProxy,
        "IRM_penalty_proxy": GroupDROIRMPenaltyProxy,
        "grad_norm_proxy": GroupDROGradNormProxy,
    }
    the_class = class_dict.get(model_type, None)
    if the_class is None:
        raise ValueError("model_type not defined in group_robust_model")
    return the_class


class GroupDROModel(TorchModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.group_weights = self.init_group_weights(
            num_groups=self.config_dict.get("num_groups")
        )

    def init_group_weights(self, num_groups=None):
        """
        Initializes the langrange multipliers
        """
        if num_groups is None:
            raise ValueError("num_groups must be provided")

        return (
            torch.ones(
                num_groups, dtype=torch.float, device=self.device, requires_grad=False,
            )
            / num_groups
        )

    def get_default_config(self):
        config_dict = super().get_default_config()
        update_dict = {
            "num_hidden": 1,
            "hidden_dim": 128,
            "drop_prob": 0.0,
            "normalize": False,
            "sparse": True,
            "sparse_mode": "csr",
            "resnet": False,
            "lr_lambda": 1e-1,
            "num_groups": 2,
            "multiplier_bound": 1,
            "use_exact_constraints": True,
            "track_running_label_counts": False,
            "update_lambda_on_val": False,
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

    def constraint_metric(self, outputs, labels, sample_weight=None):
        """
            Defines a differentiable metric that will be used to construct the constraints during training
        """
        return weighted_cross_entropy_loss(outputs, labels, sample_weight=sample_weight)

    def constraint_metric_exact(self, outputs, labels, sample_weight=None):
        """
            Defines the metric used to update the Lagrange multipliers
            Does not necessarily need to be differentiable
        """
        return weighted_cross_entropy_loss(outputs, labels, sample_weight=sample_weight)

    def get_loss_names(self):
        return ["loss"]

    def forward_on_batch(self, the_data):
        loss_dict_batch = {}

        inputs, labels, group = (
            the_data["features"],
            the_data["labels"],
            the_data["group"],
        )

        if self.config_dict.get("weighted_loss"):
            sample_weight = the_data.get("weights")
            if sample_weight is None:
                raise ValueError("weighted_loss is True, but no weights provided")
        else:
            sample_weight = None

        outputs = self.model(inputs)

        constraint_values = self.compute_constraints(
            outputs=outputs,
            labels=labels,
            group=group,
            sample_weight=sample_weight,
            exact_constraints=False,
        )

        if self.config_dict.get("use_exact_constraints"):
            constraint_values_exact = self.compute_constraints(
                outputs=outputs,
                labels=labels,
                group=group,
                sample_weight=sample_weight,
                exact_constraints=True,
            )

        # Update the group weights
        if (
            self.is_training() and not self.config_dict.get("update_lambda_on_val")
        ) or (not self.is_training() and self.config_dict.get("update_lambda_on_val")):

            constraint_values_lambda = (
                constraint_values_exact
                if self.config_dict.get("use_exact_constraints")
                else constraint_values
            )

            self.update_group_weights(
                constraint_values_lambda,
                additive_update=self.config_dict.get("additive_update"),
            )

        if self.config_dict.get("additive_update"):
            group_weights_exp = torch.exp(self.group_weights)
            # (TODO) implement norm projection for additive updates
            loss_dict_batch["loss"] = torch.dot(group_weights_exp, constraint_values)
        else:
            loss_dict_batch["loss"] = torch.dot(self.group_weights, constraint_values)

        return loss_dict_batch, outputs

    def compute_constraints(
        self, outputs, labels, group, sample_weight=None, exact_constraints=False,
    ):
        if exact_constraints:
            constraint_metric_fn = self.constraint_metric_exact
        else:
            constraint_metric_fn = self.constraint_metric

        # Initialize a tensor to hold the value of the constraint metric for each group
        constraint_values = self.get_constraint_baselines(
            exact_constraints=exact_constraints
        )

        # Initialize a tensor to hold the value of the constraints
        for the_group in group.unique():
            the_group = int(the_group.item())

            # Subset the outputs and labels
            outputs_group = outputs[group == the_group]
            labels_group = labels[group == the_group]

            if self.config_dict.get("weighted_loss"):
                sample_weight_group = sample_weight[group == the_group]
            else:
                sample_weight_group = None

            # Compute the value of the constraint metric for each group
            try:
                constraint_values[the_group] = constraint_values[the_group] + constraint_metric_fn(
                    outputs_group, labels_group, sample_weight=sample_weight_group
                )

            except MetricUndefinedError:
                logging.debug("Warning: metric undefined")
                continue
        constraint_values = constraint_values.reshape(-1)

        return constraint_values

    def get_constraint_baselines(self, exact_constraints=False):
        return torch.zeros(self.config_dict.get("num_groups")).to(self.device)

    def update_group_weights(self, constraint_values, additive_update=False):
        """
            Updates the group weights multipliers
        """
        if self.config_dict.get("additive_update"):
            self.group_weights = self.group_weights + (
                self.config_dict.get("lr_lambda")
                * torch.exp(self.group_weights)
                * constraint_values.detach()
            )
        else:
            self.group_weights = self.group_weights * torch.exp(
                self.config_dict.get("lr_lambda") * constraint_values.detach()
            )
            multiplier_sum = self.group_weights.sum()
            if multiplier_sum > self.config_dict.get("multiplier_bound"):
                self.group_weights = (
                    self.config_dict.get("multiplier_bound", 1)
                    * self.group_weights
                    / multiplier_sum
                )
            logging.debug("Lambda: {}".format(self.group_weights))
            logging.debug("Lambda sum: {}".format(self.group_weights.sum()))
