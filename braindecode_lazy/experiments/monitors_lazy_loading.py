from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np
import psutil
import time
import os

from braindecode.datautil.iterators import _compute_start_stop_block_inds


# this monitor was taken from braindecode and adapted to work with tensors as targets:
# call to ndim of ndarray was replaced by len(shape)
class LazyMisclassMonitor(object):
    """
    Monitor the examplewise misclassification rate.

    Parameters
    ----------
    col_suffix: str, optional
        Name of the column in the monitoring output.
    threshold_for_binary_case: bool, optional
        In case of binary classification with only one output prediction
        per target, define the threshold for separating the classes, i.e.
        0.5 for sigmoid outputs, or np.log(0.5) for log sigmoid outputs
    """

    def __init__(self, col_suffix='misclass', threshold_for_binary_case=None):
        self.col_suffix = col_suffix
        self.threshold_for_binary_case = threshold_for_binary_case

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        all_pred_labels = []
        all_target_labels = []
        for i_batch in range(len(all_batch_sizes)):
            preds = all_preds[i_batch]
            # preds could be examples x classes x time
            # or just
            # examples x classes
            # make sure not to remove first dimension if it only has size one
            if preds.ndim > 1:
                only_one_row = preds.shape[0] == 1

                pred_labels = np.argmax(preds, axis=1).squeeze()
                # add first dimension again if needed
                if only_one_row:
                    pred_labels = pred_labels[None]
            else:
                assert self.threshold_for_binary_case is not None, (
                    "In case of only one output, please supply the "
                    "threshold_for_binary_case parameter")
                # binary classification case... assume logits
                pred_labels = np.int32(preds > self.threshold_for_binary_case)
            # now examples x time or examples
            all_pred_labels.extend(pred_labels)
            targets = all_targets[i_batch]
            targets_ndim = len(targets.shape)
            if targets_ndim > pred_labels.ndim:
                # targets may be one-hot-encoded
                targets = np.argmax(targets, axis=1)
            elif targets_ndim < pred_labels.ndim:
                # targets may not have time dimension,
                # in that case just repeat targets on time dimension
                extra_dim = pred_labels.ndim - 1
                targets = np.repeat(np.expand_dims(targets, extra_dim),
                                    pred_labels.shape[extra_dim],
                                    extra_dim)
            assert targets.shape == pred_labels.shape
            all_target_labels.extend(targets)
        all_pred_labels = np.array(all_pred_labels)
        all_target_labels = np.array(all_target_labels)
        assert all_pred_labels.shape == all_target_labels.shape

        misclass = 1 - np.mean(all_target_labels == all_pred_labels)
        column_name = "{:s}_{:s}".format(setname, self.col_suffix)
        return {column_name: float(misclass)}


class RAMMonitor(object):
    def __init__(self):
        pass

    def monitor_epoch(self):
        pass

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        out = {}
        process = psutil.Process(os.getpid())
        usage = process.memory_info().rss
        out.update({"RAM usage": usage/1000000000})
        return out


class RMSEMonitor(object):
    """
    Compute trialwise misclasses from predictions for crops for non-dense predictions.

    Parameters
    ----------
    input_time_length: int
        Temporal length of one input to the model.
    """

    def __init__(self, input_time_length, n_preds_per_input):
        self.input_time_length = input_time_length
        self.n_preds_per_input = n_preds_per_input

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        """Assuming one hot encoding for now"""
        preds_per_trial = compute_preds_per_trial(
            all_preds, dataset, input_time_length=self.input_time_length,
            n_stride=self.n_preds_per_input)

        mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in preds_per_trial]
        mean_preds_per_trial = np.array(mean_preds_per_trial).reshape(-1)

        out = {}
        y_orig = dataset.get_targets(standardize=False)
        mean_y, std_y = np.mean(y_orig), np.std(y_orig)
        preds = (mean_preds_per_trial * std_y) + mean_y
        mse_rec = mean_squared_error(y_pred=preds, y_true=y_orig)
        rmse_rec = np.sqrt(mse_rec)
        out.update({"{}_rmse_rec".format(setname): rmse_rec})
        return out


# this monitor was taken from robintibor auto-eeg-diagnosis-example
class CroppedDiagnosisMonitor(object):
    """
    Compute trialwise misclasses from predictions for crops for non-dense predictions.
    Parameters
    ----------
    input_time_length: int
        Temporal length of one input to the model.
    """

    def __init__(self, input_time_length, n_preds_per_input):
        self.input_time_length = input_time_length
        self.n_preds_per_input = n_preds_per_input

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        """Assuming one hot encoding for now"""
        preds_per_trial = compute_preds_per_trial(
            all_preds, dataset, input_time_length=self.input_time_length,
            n_stride=self.n_preds_per_input)

        mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                preds_per_trial]
        mean_preds_per_trial = np.array(mean_preds_per_trial)

        pred_labels_per_trial = np.argmax(mean_preds_per_trial, axis=1)
        assert pred_labels_per_trial.shape == dataset.y.shape
        accuracy = np.mean(pred_labels_per_trial == dataset.y)
        misclass = 1 - accuracy
        column_name = "{:s}_misclass".format(setname)
        out = {column_name: float(misclass)}
        y = dataset.y

        n_true_positive = np.sum((y == 1) & (pred_labels_per_trial == 1))
        n_positive = np.sum(y == 1)
        if n_positive > 0:
            sensitivity = n_true_positive / float(n_positive)
        else:
            sensitivity = np.nan
        column_name = "{:s}_sensitivity".format(setname)
        out.update({column_name: float(sensitivity)})

        n_true_negative = np.sum((y == 0) & (pred_labels_per_trial == 0))
        n_negative = np.sum(y == 0)
        if n_negative > 0:
            specificity = n_true_negative / float(n_negative)
        else:
            specificity = np.nan
        column_name = "{:s}_specificity".format(setname)
        out.update({column_name: float(specificity)})
        if (n_negative > 0) and (n_positive > 0):
            auc = roc_auc_score(y, mean_preds_per_trial[:,1])
        else:
            auc = np.nan
        column_name = "{:s}_auc".format(setname)
        out.update({column_name: float(auc)})
        return out


class CroppedAgeRegressionDiagnosisMonitor(object):
    """
    Compute trialwise misclasses from predictions for crops for non-dense predictions.

    Parameters
    ----------
    input_time_length: int
        Temporal length of one input to the model.
    """

    def __init__(self, input_time_length, n_preds_per_input):
        self.input_time_length = input_time_length
        self.n_preds_per_input = n_preds_per_input

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        """Assuming one hot encoding for now"""
        preds_per_trial = compute_preds_per_trial(
            all_preds, dataset, input_time_length=self.input_time_length,
            n_stride=self.n_preds_per_input)

        mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                preds_per_trial]
        mean_preds_per_trial = np.array(mean_preds_per_trial).reshape(-1)
        #print(mean_preds_per_trial)
        #print(mean_preds_per_trial.max() - mean_preds_per_trial.min())
        y = dataset.get_targets()
        #print(y)
        #for pred_true in zip(mean_preds_per_trial, y):
        #    print(pred_true)
        assert mean_preds_per_trial.shape == y.shape
        mse = mean_squared_error(y_pred=mean_preds_per_trial, y_true=y)
        rmse = np.sqrt(mse)
        
        out = {}
        #column_name = "{:s}_mse".format(setname)
        #out = {column_name: float(mse)}
        column_name = "{:s}_rmse".format(setname)
        out.update({column_name: float(rmse)})
        return out


def compute_preds_per_trial(preds_per_batch, dataset, input_time_length,
                            n_stride):
    n_trials = len(dataset.X)
    i_pred_starts = [input_time_length -
                     n_stride] * n_trials
    i_pred_stops = [t.shape[1] for t in dataset.X]

    start_stop_block_inds_per_trial = _compute_start_stop_block_inds(
        i_pred_starts,
        i_pred_stops, input_time_length, n_stride,
        False)

    n_rows_per_trial = [len(block_inds) for block_inds in
                        start_stop_block_inds_per_trial]

    all_preds_arr = np.concatenate(preds_per_batch, axis=0)
    i_row = 0
    preds_per_trial = []
    for n_rows in n_rows_per_trial:
        preds_per_trial.append(all_preds_arr[i_row:i_row + n_rows])
        i_row += n_rows
    assert i_row == len(all_preds_arr)
    return preds_per_trial
