"""
Python classes that help with model evaluation 
Based on Stephen Pfohl's evaluation fns
"""
import numpy as np
import pandas as pd

import warnings
import scipy

from collections import ChainMap
from sklearn.calibration import CalibratedClassifierCV as ccv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score, 
    average_precision_score, 
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    brier_score_loss,
    log_loss
)


def df_dict_concat(df_dict, outer_index_name="outer_index", drop_outer_index=False):
    """
    Concatenate a dictionary of dataframes together and remove the inner index
    """
    if isinstance(outer_index_name, str):
        reset_level = 1
    else:
        reset_level = len(outer_index_name)

    return (
        pd.concat(df_dict, sort=False)
        .reset_index(level=reset_level, drop=True)
        .rename_axis(outer_index_name)
        .reset_index(drop=drop_outer_index)
    )

"""
Metrics for StandardEvaluator
"""
def try_metric_fn(*args, metric_fn=None, default_value=-1, **kwargs):
    """
    Tries to call a metric function, returns default_value if fails
    """
    if metric_fn is None:
        raise ValueError("Must provide metric_fn")
    try:
        return metric_fn(*args, **kwargs)
    except ValueError:
        warnings.warn("Error in metric_fn, filling with default_value")
        return default_value

    
def try_roc_auc_score(*args, **kwargs):
    return try_metric_fn(*args, metric_fn=roc_auc_score, default_value=-1, **kwargs)


def try_log_loss(*args, **kwargs):
    return try_metric_fn(*args, metric_fn=log_loss, default_value=1e18, **kwargs)

def compute_threshold_metric(y_true, pred_prob, threshold, metric, **kwargs):
    """
    Threshold metrics for binary prediction tasks
    """
    y_pred = 1*(pred_prob>threshold)
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    
    metric = metric.lower()
    
    metric_dict = {}
    
    # Sensitivity, hit rate, recall, or true positive rate
    metric_dict['sensitivity'] = TP/(TP+FN)
    # Specificity or true negative rate
    metric_dict['specificity'] = TN/(TN+FP) 
    # Precision or positive predictive value
    metric_dict['ppv'] = TP/(TP+FP)
    # Negative predictive value
    metric_dict['npv'] = TN/(TN+FN)
    # Fall out or false positive rate
    metric_dict['fpr'] = FP/(FP+TN)
    # False negative rate
    metric_dict['fnr'] = FN/(TP+FN)
    # False discovery rate
    metric_dict['fdr'] = FP/(TP+FP)
    
    return metric_dict[metric]

"""
Calibration Metrics
"""
def absolute_calibration_error(*args, **kwargs):
    evaluator = CalibrationEvaluator()
    return evaluator.absolute_calibration_error(*args, **kwargs)


def relative_calibration_error(*args, **kwargs):
    evaluator = CalibrationEvaluator()
    return evaluator.relative_calibration_error(*args, **kwargs)


def expected_calibration_error(
    labels, pred_probs, num_bins=10, metric_variant="abs", quantile_bins=False
):
    """
        Computes the calibration error with a binning estimator over equal sized bins
        See http://arxiv.org/abs/1706.04599 and https://arxiv.org/abs/1904.01685.
        Does not currently support sample weights
    """
    if metric_variant == "abs":
        transform_func = np.abs
    elif (metric_variant == "squared") or (metric_variant == "rmse"):
        transform_func = np.square
    elif metric_variant == "signed":
        transform_func = identity
    else:
        raise ValueError("provided metric_variant not supported")

    if quantile_bins:
        cut_fn = pd.qcut
    else:
        cut_fn = pd.cut

    bin_ids = cut_fn(pred_probs, num_bins, labels=False, retbins=False)
    df = pd.DataFrame({"pred_probs": pred_probs, "labels": labels, "bin_id": bin_ids})
    ece_df = (
        df.groupby("bin_id")
        .agg(
            pred_probs_mean=("pred_probs", "mean"),
            labels_mean=("labels", "mean"),
            bin_size=("pred_probs", "size"),
        )
        .assign(
            bin_weight=lambda x: x.bin_size / df.shape[0],
            err=lambda x: transform_func(x.pred_probs_mean - x.labels_mean),
        )
    )
    result = np.average(ece_df.err.values, weights=ece_df.bin_weight)
    if metric_variant == "rmse":
        result = np.sqrt(result)
    return result


"""
Evaluators:
- StandardEvaluator
- CalibrationEvaluator
"""
class StandardEvaluator:
    def __init__(self, metrics=None, threshold_metrics=None):
        if metrics is None:
            self.metrics = self.get_default_threshold_free_metrics()
        else:
            self.metrics = metrics
        
        # Threshold metrics will only compute if threshold_var exists
        # in input dataframe. Note that threshold metrics currently
        # do not support sample weights
        if threshold_metrics is None:
            self.threshold_metrics = self.get_default_threshold_metrics()
        else:
            self.threshold_metrics = threshold_metrics

    def evaluate(
        self,
        df,
        strata_vars=None,
        result_name="performance",
        weight_var=None,
        label_var="labels",
        pred_prob_var="pred_probs",
        threshold_var="threshold",
    ):
        """
        Evaluates predictions against a set of labels with a set of metric functions
        Arguments:
            df: a dataframe with one row per prediction
            result_name: a string that will be used to label the metric values in the result
            weight_var: a string identifier for sample weights in df
            label_var: a string identifier for the outcome labels in df
            pred_prob_var: a string identifier for the predicted probabilities in df
            threshold_var: a string identifier for the probability threshold in df
        """
        # Threshold free metrics
        metric_fns = self.get_threshold_free_metrics(
            metrics=self.metrics
        )
        
        if strata_vars is not None:
            strata_vars = [var for var in strata_vars if var in df.columns]
        if (strata_vars is None) or (len(strata_vars) == 0):
            result_df = (
                pd.DataFrame(
                    {
                        metric: metric_fn(
                            df[label_var].values, df[pred_prob_var].values
                        )
                        if weight_var is None
                        else metric_fn(
                            df[label_var].values,
                            df[pred_prob_var].values,
                            sample_weight=df[weight_var].values,
                        )
                        for metric, metric_fn in metric_fns.items()
                    },
                    index=[result_name],
                )
                .transpose()
                .rename_axis("metric")
                .reset_index()
            )
        else:
            result_df = df_dict_concat(
                {
                    metric: df.groupby(strata_vars)
                    .apply(
                        lambda x: metric_func(
                            x[label_var].values, x[pred_prob_var].values
                        )
                        if weight_var is None
                        else metric_func(
                            x[label_var].values,
                            x[pred_prob_var].values,
                            sample_weight=x[weight_var].values,
                        )
                    )
                    .rename(index=result_name)
                    .rename_axis(strata_vars)
                    .reset_index()
                    for metric, metric_func in metric_fns.items()
                },
                "metric",
            )
        
        ## Threshold metrics
        if threshold_var in df.columns:
            
            metric_fns = self.get_threshold_metric(
                metrics = self.threshold_metrics,
            )
            
            # do threshold metric
            if strata_vars is not None:
                strata_vars = [var for var in strata_vars if var in df.columns]
            if (strata_vars is None) or (len(strata_vars) == 0):
                result_df_thresh = (
                    pd.DataFrame(
                        {
                            metric: metric_fn(
                                df[label_var].values, 
                                df[pred_prob_var].values, 
                                df[threshold_var].values
                            )
                            
                            for metric, metric_fn in metric_fns.items()
                        },
                        index=[result_name],
                    )
                    .transpose()
                    .rename_axis("metric")
                    .reset_index()
                )
            else:
                result_df_thresh = df_dict_concat(
                    {
                        metric: df.groupby(strata_vars)
                        .apply(
                            lambda x: metric_func(
                                x[label_var].values, 
                                x[pred_prob_var].values,
                                x[threshold_var].values
                            )
                        )
                        .rename(index=result_name)
                        .rename_axis(strata_vars)
                        .reset_index()
                        for metric, metric_func in metric_fns.items()
                    },
                    "metric",
                )
                
            result_df = pd.concat(
                (result_df, result_df_thresh), 
                axis = 0,
                ignore_index = True
            )
            
        return result_df

    def get_default_threshold_free_metrics(self):
        """
        Defines the string identifiers for the default threshold free metrics
        """
        return [
            "auc",
            "auprc",
            "loss_bce",
            "ace_rmse_logistic_log",
            "ace_abs_logistic_log",
        ]
    
    def get_default_threshold_metrics(self):
        """
        Defines the string identifiers for the default threshold metrics
        """
        return [
            "ppv",
            "npv",
            "sensitivity",
            "specificity",
        ]
    
    def get_threshold_metric(self, metrics=None):
        """
        Defines the set of allowable threshold metric functions
        """
        base_metric_dict={
            "ppv": lambda *args, **kwargs: compute_threshold_metric(
                *args, metric='ppv', **kwargs
            ),
            "npv": lambda *args, **kwargs: compute_threshold_metric(
                *args, metric='npv', **kwargs
            ),
            "sensitivity": lambda *args, **kwargs: compute_threshold_metric(
                *args, metric='sensitivity', **kwargs
            ),
            "specificity": lambda *args, **kwargs: compute_threshold_metric(
                *args, metric='specificity', **kwargs
            ),
            "fpr": lambda *args, **kwargs: compute_threshold_metric(
                *args, metric='fpr', **kwargs
            ),
            "fnr": lambda *args, **kwargs: compute_threshold_metric(
                *args, metric='fnr', **kwargs
            ),
            "fdr": lambda *args, **kwargs: compute_threshold_metric(
                *args, metric='fdr', **kwargs
            ),
        }
        
        if metrics is None:
            return base_metric_dict
        else:
            return {
                key: base_metric_dict[key]
                for key in metrics
                if key in base_metric_dict.keys()
            }

    def get_threshold_free_metrics(self, metrics=None):
        """
        Defines the set of allowable threshold free metric functions
        """
        base_metric_dict = {
            "auc": try_roc_auc_score,
            "auprc": average_precision_score,
            "brier": brier_score_loss,
            "loss_bce": try_log_loss,
            "cox_slope": lambda *args, **kwargs: cox_calibration(
                *args, out = 'slope', **kwargs
            ),
            "cox_intercept": lambda *args, **kwargs: cox_calibration(
                *args, out = 'intercept', **kwargs
            ),
            "ece_q_abs": lambda *args, **kwargs: expected_calibration_error(
                *args, metric_variant="abs", quantile_bins=True, **kwargs
            ),
            "ece_q_rmse": lambda *args, **kwargs: expected_calibration_error(
                *args, metric_variant="rmse", quantile_bins=True, **kwargs
            ),
            "ece_abs": lambda *args, **kwargs: expected_calibration_error(
                *args, metric_variant="abs", quantile_bins=False, **kwargs
            ),
            "ece_rmse": lambda *args, **kwargs: expected_calibration_error(
                *args, metric_variant="rmse", quantile_bins=False, **kwargs
            ),
            "ace_abs_logistic_log": lambda *args, **kwargs: absolute_calibration_error(
                *args,
                metric_variant="abs",
                model_type="logistic",
                transform="log",
                **kwargs,
            ),
            "ace_abs_bin_log": lambda *args, **kwargs: absolute_calibration_error(
                *args, metric_variant="abs", model_type="bin", transform="log", **kwargs
            ),
            "ace_rmse_logistic_log": lambda *args, **kwargs: absolute_calibration_error(
                *args,
                metric_variant="rmse",
                model_type="logistic",
                transform="log",
                **kwargs,
            ),
            "ace_rmse_bin_log": lambda *args, **kwargs: absolute_calibration_error(
                *args,
                metric_variant="rmse",
                model_type="bin",
                transform="log",
                **kwargs,
            ),
            "ace_signed_logistic_log": lambda *args, **kwargs: absolute_calibration_error(
                *args,
                metric_variant="signed",
                model_type="logistic",
                transform="log",
                **kwargs,
            ),
            "ace_signed_bin_log": lambda *args, **kwargs: absolute_calibration_error(
                *args,
                metric_variant="signed",
                model_type="bin",
                transform="log",
                **kwargs,
            ),
            "ace_abs_logistic_none": lambda *args, **kwargs: absolute_calibration_error(
                *args,
                metric_variant="abs",
                model_type="logistic",
                transform=None,
                **kwargs,
            ),
            "ace_abs_bin_none": lambda *args, **kwargs: absolute_calibration_error(
                *args, metric_variant="abs", model_type="bin", transform=None, **kwargs
            ),
            "ace_rmse_logistic_none": lambda *args, **kwargs: absolute_calibration_error(
                *args,
                metric_variant="rmse",
                model_type="logistic",
                transform=None,
                **kwargs,
            ),
            "ace_rmse_bin_none": lambda *args, **kwargs: absolute_calibration_error(
                *args,
                metric_variant="rmse",
                model_type="bin",
                transform=None,
                **kwargs,
            ),
            "ace_signed_logistic_none": lambda *args, **kwargs: absolute_calibration_error(
                *args,
                metric_variant="signed",
                model_type="logistic",
                transform=None,
                **kwargs,
            ),
            "ace_signed_bin_none": lambda *args, **kwargs: absolute_calibration_error(
                *args,
                metric_variant="signed",
                model_type="bin",
                transform=None,
                **kwargs,
            ),
            "spielhalter_z": lambda *args, **kwargs: spiegelhalter_z(
                *args, **kwargs
            ),
            "mean_prediction": lambda *args, **kwargs: mean_prediction(
                *args, the_label=None, **kwargs
            ),
            "mean_prediction_0": lambda *args, **kwargs: mean_prediction(
                *args, the_label=0, **kwargs
            ),
            "mean_prediction_1": lambda *args, **kwargs: mean_prediction(
                *args, the_label=1, **kwargs
            ),
        }
        if metrics is None:
            return base_metric_dict
        else:
            return {
                key: base_metric_dict[key]
                for key in metrics
                if key in base_metric_dict.keys()
            }

    def bootstrap_evaluate(
        self,
        df,
        n_boot=1000,
        strata_vars=None,
        result_name="performance",
        weight_var=None,
        label_var="labels",
        pred_prob_var="pred_probs",
        **kwargs
    ):
        """
        Applies the quantile bootstrap to the evaluation procedure
        """

        result_df_dict = {}

        for i in range(n_boot):

            if strata_vars is None:
                df_boot = df.sample(frac=1.0, replace=True).reset_index(drop=True)
            else:
                df_boot = (
                    df.groupby(strata_vars)
                    .sample(frac=1.0, replace=True)
                    .reset_index(drop=True)
                )

            result_df_dict[i] = self.evaluate(
                df=df_boot,
                strata_vars=strata_vars,
                result_name=result_name,
                weight_var=weight_var,
                label_var=label_var,
                pred_prob_var=pred_prob_var,
                **kwargs,
            )
        result_df = pd.concat(result_df_dict)
        result_df = (
            result_df.reset_index(level=-1, drop=True)
            .rename_axis("boot_id")
            .reset_index()
        )
        if strata_vars is None:
            strata_vars_ci = "metric"
        else:
            strata_vars_ci = ["metric"] + strata_vars

        if kwargs.get("group_var_name"):
            if isinstance(strata_vars_ci, str):
                strata_vars_ci = [strata_vars_ci] + [kwargs.get("group_var_name")]
            elif isinstance(strata_vars_ci, list):
                strata_vars_ci = strata_vars_ci + [kwargs.get("group_var_name")]

        result_df_ci = (
            result_df.groupby(strata_vars_ci)
            .apply(lambda x: np.quantile(x[result_name], [0.025, 0.5, 0.975]))
            .rename(result_name)
            .reset_index()
            .assign(
                CI_lower=lambda x: x[result_name].str[0],
                CI_med=lambda x: x[result_name].str[1],
                CI_upper=lambda x: x[result_name].str[2],
            )
            .drop(columns=[result_name])
        )

        return result_df_ci

class CalibrationEvaluator:
    """
    Evaluator that computes absolute and relative calibration errors
    """

    def get_calibration_density_df(
        self,
        labels,
        pred_probs,
        sample_weight=None,
        model_type="logistic",
        transform=None,
    ):

        model = self.init_model(model_type=model_type)

        df = pd.DataFrame({"pred_probs": pred_probs, "labels": labels})
        if sample_weight is not None:
            df = df.assign(sample_weight=sample_weight)

        valid_transforms = ["log", "c_log_log"]
        if transform is None:
            df = df.assign(model_input=lambda x: x.pred_probs)
            model_input = df.model_input.values.reshape(-1, 1)
        elif transform in valid_transforms:
            df = df.query("(pred_probs > 1e-15) & (pred_probs < (1 - 1e-15))")
            if transform == "log":
                df = df.assign(model_input=lambda x: np.log(x.pred_probs))
            elif transform == "c_log_log":
                df = df.assign(model_input=lambda x: self.c_log_log(x.pred_probs))
        else:
            raise ValueError("Invalid transform provided")
        model_input = df.model_input.values.reshape(-1, 1)
        model.fit(
            model_input,
            df.labels.values,
            sample_weight=df.sample_weight.values
            if "sample_weight" in df.columns
            else None,
        )
        calibration_density = model.predict_proba(model_input)
        if len(calibration_density.shape) > 1:
            calibration_density = calibration_density[:, -1]
        # df = df.assign(calibration_density=model.predict_proba(model_input)[:, -1])
        df = df.assign(calibration_density=calibration_density)
        return df, model

    def absolute_calibration_error(
        self,
        labels,
        pred_probs,
        sample_weight=None,
        metric_variant="abs",
        model_type="logistic",
        transform=None,
    ):

        df, model = self.get_calibration_density_df(
            labels,
            pred_probs,
            sample_weight=sample_weight,
            model_type=model_type,
            transform=transform,
        )
        if "sample_weight" in df.columns:
            sample_weight = df.sample_weight
        else:
            sample_weight = None

        if metric_variant == "squared":
            return self.weighted_mean(
                (df.calibration_density - df.pred_probs) ** 2,
                sample_weight=sample_weight,
            )
        elif metric_variant == "rmse":
            return np.sqrt(
                self.weighted_mean(
                    (df.calibration_density - df.pred_probs) ** 2,
                    sample_weight=sample_weight,
                )
            )
        elif metric_variant == "abs":
            return self.weighted_mean(
                np.abs(df.calibration_density - df.pred_probs),
                sample_weight=sample_weight,
            )
        elif metric_variant == "signed":
            return self.weighted_mean(
                df.calibration_density - df.pred_probs, sample_weight=sample_weight
            )
        else:
            raise ValueError("Invalid option specified for metric")

    def relative_calibration_error(
        self,
        labels,
        pred_probs,
        group,
        sample_weight=None,
        metric_variant="abs",
        model_type="logistic",
        transform=None,
        compute_ace=False,
        return_models=False,
        return_calibration_density=False,
    ):

        calibration_density_df_overall, model_overall = self.get_calibration_density_df(
            labels,
            pred_probs,
            sample_weight=sample_weight,
            model_type=model_type,
            transform=transform,
        )

        df = pd.DataFrame({"pred_probs": pred_probs, "labels": labels, "group": group})
        if sample_weight is not None:
            df = df.assign(sample_weight=sample_weight)

        ace_dict = {}
        rce_dict = {}
        model_dict = {}
        calibration_density_dict = {}
        for group_id, group_df in df.groupby(group):

            (
                calibration_density_dict[group_id],
                model_dict[group_id],
            ) = self.get_calibration_density_df(
                group_df.labels,
                group_df.pred_probs,
                sample_weight=group_df.sample_weight
                if "sample_weight" in group_df.columns
                else None,
                model_type=model_type,
                transform=transform,
            )

            calib_diff = (
                model_dict[group_id].predict_proba(
                    calibration_density_dict[group_id].model_input.values.reshape(
                        -1, 1
                    ),
                )[:, -1]
                - model_overall.predict_proba(
                    calibration_density_dict[group_id].model_input.values.reshape(
                        -1, 1
                    ),
                )[:, -1]
            )

            group_sample_weight = (
                calibration_density_dict[group_id].sample_weight
                if "sample_weight" in calibration_density_dict[group_id].columns
                else None
            )
            if metric_variant == "squared":
                rce_dict[group_id] = self.weighted_mean(
                    calib_diff ** 2, sample_weight=group_sample_weight
                )
            elif metric_variant == "rmse":
                rce_dict[group_id] = np.sqrt(
                    self.weighted_mean(
                        calib_diff ** 2, sample_weight=group_sample_weight
                    )
                )
            elif metric_variant == "abs":
                rce_dict[group_id] = self.weighted_mean(
                    np.abs(calib_diff), sample_weight=group_sample_weight
                )
            elif metric_variant == "signed":
                rce_dict[group_id] = self.weighted_mean(
                    calib_diff, sample_weight=group_sample_weight
                )
            else:
                raise ValueError("Invalid option specified for metric")

            if compute_ace:
                if metric_variant == "squared":
                    ace_dict[group_id] = self.weighted_mean(
                        (
                            calibration_density_dict[group_id].calibration_density
                            - calibration_density_dict[group_id].pred_probs
                        )
                        ** 2,
                        sample_weight=group_sample_weight,
                    )
                elif metric_variant == "rmse":
                    ace_dict[group_id] = np.sqrt(
                        self.weighted_mean(
                            (
                                calibration_density_dict[group_id].calibration_density
                                - calibration_density_dict[group_id].pred_probs
                            )
                            ** 2,
                            sample_weight=group_sample_weight,
                        )
                    )
                elif metric_variant == "abs":
                    ace_dict[group_id] = self.weighted_mean(
                        np.abs(
                            calibration_density_dict[group_id].calibration_density
                            - calibration_density_dict[group_id].pred_probs
                        ),
                        sample_weight=group_sample_weight,
                    )
                elif metric_variant == "signed":
                    ace_dict[group_id] = self.weighted_mean(
                        calibration_density_dict[group_id].calibration_density
                        - calibration_density_dict[group_id].pred_probs,
                        sample_weight=group_sample_weight,
                    )
                else:
                    raise ValueError("Invalid option specified for metric")
        result_dict = {}
        result_dict["result"] = (
            pd.DataFrame(rce_dict, index=["relative_calibration_error"])
            .transpose()
            .rename_axis("group")
            .reset_index()
        )
        if compute_ace:
            ace_df = (
                pd.DataFrame(ace_dict, index=["absolute_calibration_error"])
                .transpose()
                .rename_axis("group")
                .reset_index()
            )
            result_dict["result"] = result_dict["result"].merge(ace_df)
        if return_models:
            result_dict["model_dict_group"] = model_dict
            result_dict["model_overall"] = model_overall
        if return_calibration_density:
            result_dict["calibration_density_group"] = (
                pd.concat(calibration_density_dict)
                .reset_index(level=-1, drop=True)
                .rename_axis("group")
                .reset_index()
            )
            result_dict["calibration_density_overall"] = calibration_density_df_overall

        return result_dict
    
    @staticmethod
    def c_log_log(x):
        return np.log(-np.log(1 - x))

    @staticmethod
    def weighted_mean(x, sample_weight=None):
        if sample_weight is None:
            return x.mean()
        else:
            return np.average(x, weights=sample_weight)

    def init_model(self, model_type, **kwargs):
        if model_type == "logistic":
            model = LogisticRegression(
                solver="lbfgs", penalty="none", max_iter=10000, **kwargs
            )
        elif model_type == "rf":
            model = RandomForestClassifier(**kwargs)
        elif model_type == "bin":
            model = BinningEstimator(**kwargs)
        else:
            raise ValueError("Invalid model_type not provided")
        return model


class BinningEstimator:
    def __init__(self, num_bins=10, quantile_bins=True):
        self.num_bins = num_bins
        self.discretizer = KBinsDiscretizer(
            n_bins=num_bins,
            encode="ordinal",
            strategy="quantile" if quantile_bins else "uniform",
        )
        self.prob_y_lookup = -1e18 * np.ones(num_bins)

    def fit(self, x, y, sample_weight=None):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        binned_x = self.discretizer.fit_transform(x)
        binned_x = binned_x.squeeze()
        for bin_id in range(self.num_bins):
            mask = binned_x == bin_id
            if (mask).sum() == 0:
                print("No data in bin {}".format(bin_id))
            if sample_weight is None:
                self.prob_y_lookup[bin_id] = y[mask].mean()
            else:
                self.prob_y_lookup[bin_id] = np.average(
                    y[mask], weights=sample_weight[mask]
                )

    def predict_proba(self, x):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        binned_x = self.discretizer.transform(x).squeeze().astype(np.int64)
        return self.prob_y_lookup[binned_x]
