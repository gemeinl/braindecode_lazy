from sklearn.model_selection import KFold
import torch.nn.functional as F
from torch import nn
import pandas as pd
import torch as th
import numpy as np
import datetime
import logging
import json
import time
import sys
import os

from braindecode.torch_ext.schedulers import ScheduledOptimizer, CosineAnnealing
from braindecode.torch_ext.util import np_to_var, var_to_np, set_random_seeds
from braindecode.experiments.monitors import RuntimeMonitor, LossMonitor
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.models.util import to_dense_prediction_model
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.experiments.stopcriteria import MaxEpochs
from braindecode.experiments.experiment import Experiment
from braindecode.torch_ext.optimizers import AdamW
from braindecode.models.deep4 import Deep4Net

# my imports
from braindecodelazy.experiments.monitors_lazy_loading import LazyMisclassMonitor, \
    RMSEMonitor, CroppedDiagnosisMonitor, CroppedAgeRegressionDiagnosisMonitor, \
    compute_preds_per_trial
from braindecodelazy.datautil.iterators import LoadCropsFromTrialsIterator
from braindecodelazy.datasets.tuh_lazy import TuhLazy, TuhLazySubset
from braindecodelazy.datasets.tuh import Tuh, TuhSubset
from examples.utils import parse_run_args

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("test")

sys.path.insert(1, "/home/gemeinl/code/braindecodelazy/")


def nll_loss_on_mean(preds, targets):
    mean_preds = th.mean(preds, dim=2, keepdim=False)
    return F.nll_loss(mean_preds, targets)


def mse_loss_on_mean(preds, targets):
    mean_preds = th.mean(preds, dim=2, keepdim=False).squeeze()
    return F.mse_loss(mean_preds, targets)


def run_exp(train_folder,
            n_recordings,
            n_chans,
            model_name,
            n_start_chans,
            n_chan_factor,
            input_time_length,
            final_conv_length,
            model_constraint,
            stride_before_pool,
            init_lr,
            batch_size,
            max_epochs,
            cuda,
            num_workers,
            task,
            weight_decay,
            seed,
            n_folds,
            shuffle_folds,
            lazy_loading,
            eval_folder,
            result_folder,
            ):
    logging.info("Targets for this task: <{}>".format(task))

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    if task == "age":
        loss_function = mse_loss_on_mean
        remember_best_column = "valid_rmse"
        n_classes = 1
    else:
        loss_function = nll_loss_on_mean
        remember_best_column = "valid_misclass"
        n_classes = 2

    if model_constraint is not None:
        assert model_constraint == 'defaultnorm'
        model_constraint = MaxNormDefaultConstraint()

    stop_criterion = MaxEpochs(max_epochs)

    set_random_seeds(seed=seed, cuda=cuda)
    if model_name == 'shallow':
        model = ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes,
                                n_filters_time=n_start_chans,
                                n_filters_spat=n_start_chans,
                                input_time_length=input_time_length,
                                final_conv_length=final_conv_length).create_network()
    elif model_name == 'deep':
        model = Deep4Net(n_chans, n_classes,
                         n_filters_time=n_start_chans,
                         n_filters_spat=n_start_chans,
                         input_time_length=input_time_length,
                         n_filters_2=int(n_start_chans * n_chan_factor),
                         n_filters_3=int(n_start_chans * (n_chan_factor ** 2.0)),
                         n_filters_4=int(n_start_chans * (n_chan_factor ** 3.0)),
                         final_conv_length=final_conv_length,
                         stride_before_pool=stride_before_pool).create_network()

    else:
        assert False, "unknown model name {:s}".format(model_name)

    if task == "age":
        # remove softmax layer, set n_classes to 1
        model.n_classes = 1
        new_model = nn.Sequential()
        for name, module_ in model.named_children():
            if name == "softmax":
                continue
            new_model.add_module(name, module_)
        model = new_model

    if cuda:
        model.cuda()

    to_dense_prediction_model(model)
    logging.info("Model:\n{:s}".format(str(model)))

    test_input = np_to_var(np.ones((2, n_chans, input_time_length, 1),
                                   dtype=np.float32))
    if list(model.parameters())[0].is_cuda:
        test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]

    if eval_folder is None:
        logging.info("will do validation")
        if lazy_loading:
            dataset = TuhLazy(train_folder, target=task,
                              n_recordings=n_recordings, extension=".h5")
        else:
            dataset = Tuh(train_folder, n_recordings=n_recordings,
                          target=task, extension=".h5")

        rest = seed % n_folds
        indices = np.arange(len(dataset))
        kf = KFold(n_splits=n_folds, shuffle=shuffle_folds)
        for i, (train_ind, test_ind) in enumerate(kf.split(indices)):
            assert len(np.intersect1d(train_ind, test_ind)) == 0, \
                "train and test set overlap!"

            if n_folds - i == rest:
                break

        if lazy_loading:
            logging.info("using lazy loading")
            test_subset = TuhLazySubset(dataset, test_ind)
            train_subset = TuhLazySubset(dataset, train_ind)

        else:
            logging.info("using traditional loading")
            test_subset = TuhSubset(dataset, test_ind)
            train_subset = TuhSubset(dataset, train_ind)
    else:
        logging.info("will do final evaluation")
        if lazy_loading:
            train_subset = TuhLazy(train_folder, target=task, extension=".h5")
            test_subset = TuhLazy(eval_folder, target=task, extension=".h5")
        else:
            train_subset = Tuh(train_folder, target=task, extension=".h5")
            test_subset = Tuh(eval_folder, target=task, extension=".h5")

    if lazy_loading:
        iterator = LoadCropsFromTrialsIterator(
            input_time_length, n_preds_per_input, batch_size,
            seed=seed, num_workers=num_workers)
    else:
         iterator = CropsFromTrialsIterator(batch_size, input_time_length,
                                            n_preds_per_input, seed)

    monitors = []
    monitors.append(LossMonitor())
    monitors.append(RuntimeMonitor())
    if task == "age":
        monitors.append(CroppedAgeRegressionDiagnosisMonitor(input_time_length, n_preds_per_input))
        monitors.append(RMSEMonitor(input_time_length, n_preds_per_input))
    else:
        monitors.append(LazyMisclassMonitor(col_suffix='sample_misclass'))
        monitors.append(CroppedDiagnosisMonitor(input_time_length, n_preds_per_input))

    n_updates_per_epoch = sum([1 for _ in iterator.get_batches(train_subset, shuffle=True)])
    n_updates_per_period = n_updates_per_epoch * max_epochs

    adamw = AdamW(model.parameters(), init_lr, weight_decay=weight_decay)
    scheduler = CosineAnnealing(n_updates_per_period)
    optimizer = ScheduledOptimizer(scheduler, adamw, schedule_weight_decay=True)

    exp = Experiment(model,
                     train_subset,
                     None,
                     test_subset,
                     iterator,
                     loss_function,
                     optimizer,
                     model_constraint,
                     monitors,
                     stop_criterion,
                     remember_best_column=remember_best_column,
                     run_after_early_stop=False,
                     batch_modifier=None,
                     cuda=cuda,
                     do_early_stop=False,
                     reset_after_second_run=False, )
    exp.run()
    return exp


def write_kwargs_and_epochs_dfs(kwargs, exp):
    result_folder = kwargs["result_folder"]
    if result_folder is None:
        return
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    with open(result_folder + "config.json", "w") as json_file:
        json.dump(kwargs, json_file, indent=4, sort_keys=True)
    exp.epochs_df.to_csv(result_folder + "epochs_df_" + str(kwargs["seed"]) + ".csv")


def write_predictions(y, mean_preds_per_trial, setname, kwargs, exp):
    result_folder = kwargs["result_folder"]
    if result_folder is None:
        return
    logging.info("length of y is {}".format(len(y)))
    logging.info("length of mean_preds_per_trial is {}".format(len(mean_preds_per_trial)))
    assert len(y) == len(mean_preds_per_trial)

    if kwargs["task"] == "pathological":
        column0, column1 = "non-pathological", "pathological"
        a_dict = {column0: mean_preds_per_trial[:, 0],
                  column1: mean_preds_per_trial[:, 1],
                  "true_pathological": y}
    elif kwargs["task"] == "gender":
        column0, column1 = "male", "female"
        a_dict = {column0: mean_preds_per_trial[:, 0],
                  column1: mean_preds_per_trial[:, 1],
                  "true_gender": y}
    elif kwargs["task"] == "age":
        # very ugly to access a monitor in a hardcoded location
        mean_train_age = exp.monitors[-1].mean_train_age
        std_train_age = exp.monitors[-1].std_train_age
        # recreate actual age from standardized age
        y = (y * std_train_age) + mean_train_age
        mean_preds_per_trial = (std_train_age * mean_preds_per_trial) + mean_train_age

        column = "age"
        y_pred = mean_preds_per_trial.squeeze()
        a_dict = {column: y_pred, "true_age": y}
    # store predictions
    pd.DataFrame.from_dict(a_dict).to_csv(result_folder + "predictions_" + setname +
                                          "_" + str(kwargs["seed"]) + ".csv")


def make_final_predictions(kwargs, exp):
    test_input = np_to_var(np.ones((2, kwargs["n_chans"], kwargs["input_time_length"], 1), dtype=np.float32))
    if list(exp.model.parameters())[0].is_cuda:
        test_input = test_input.cuda()
    out = exp.model(test_input)

    exp.model.eval()
    for setname in ('train', 'test'):
        dataset = exp.datasets[setname]
        if kwargs["cuda"]:
            preds_per_batch = [var_to_np(exp.model(np_to_var(b[0]).cuda()))
                               for b in exp.iterator.get_batches(dataset, shuffle=False)]
        else:
            preds_per_batch = [var_to_np(exp.model(np_to_var(b[0])))
                               for b in exp.iterator.get_batches(dataset, shuffle=False)]
        preds_per_trial = compute_preds_per_trial(
            preds_per_batch, dataset,
            input_time_length=exp.iterator.input_time_length,
            n_stride=exp.iterator.n_preds_per_input)
        mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                preds_per_trial]
        mean_preds_per_trial = np.array(mean_preds_per_trial)

        write_predictions(dataset.y, mean_preds_per_trial, setname, kwargs, exp)


def main():
    logging.basicConfig(level=logging.DEBUG)
    kwargs = parse_run_args()
    start_time = time.time()
    exp = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    logging.info("Experiment runtime: {:.2f} sec".format(run_time))

    write_kwargs_and_epochs_dfs(kwargs, exp)
    make_final_predictions(kwargs, exp)


if __name__ == '__main__':
    main()
