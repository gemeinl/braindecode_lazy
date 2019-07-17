from itertools import product
import pandas as pd
import numpy as np
import argparse
import logging
import os


def dfs_from_dir(path, prefix="epochs_df_"):
    last_row_df = pd.DataFrame()
    i = 0
    while True:
        df_file = path + prefix + str(i) + ".csv"
        if not os.path.exists(df_file):
            return last_row_df
        df = pd.read_csv(df_file, index_col=0)
        last_row = df.iloc[-1]
        last_row["epoch"] = len(df) - 1
        last_row_df = last_row_df.append(last_row, ignore_index=True)
        i += 1


def df_list_from_dir(path, prefix="epochs_df_"):
    df = []
    i = 0
    while True:
        df_file = path + prefix + str(i) + ".csv"
        if not os.path.exists(df_file):
            return df
        df.append(pd.read_csv(df_file, index_col=0))
        i += 1


# all_preds_df = concat_all_preds(
#     "/data/schirrmr/gemeinl/results/networks/{}/{}/{}/predictions_test_{}.csv",
#     "deep", "cv", 5)
def concat_all_preds(path, model, split, n):
    all_preds_df = pd.DataFrame()
    for i in range(n):
        f_path = path.format(model, "{}", split, i)
        patho_df = pd.read_csv(f_path.format("pathological"), index_col=0)
        gender_df = pd.read_csv(f_path.format("gender"), index_col=0)
        age_df = pd.read_csv(f_path.format("age"), index_col=0)
        assert len(patho_df) == len(gender_df) == len(
            age_df), "prediction dfs are of unequal length"
        merged_df = pd.concat([patho_df, gender_df, age_df], axis=1)
        merged_df["id"] = pd.Series(len(merged_df) * [i])
        all_preds_df = all_preds_df.append(merged_df)

    return all_preds_df


def read_network_results(directory, models, decoding_tasks):
    result_df = pd.DataFrame()
    for model, decoding_task, decoding_type in product(
            models, decoding_tasks, ["cv", "eval"]):
        curr_path = os.path.join(
            directory, model, decoding_task, decoding_type, "")
        if os.path.exists(curr_path):
            dfs = df_list_from_dir(curr_path)
            rmse = np.nan
            misclass = np.nan
            misclass_or_rmse = "misclass" if "train_misclass" in dfs[0]\
                else "rmse"
            result = np.mean([df["test_"+misclass_or_rmse].iloc[-1]
                              for df in dfs], axis=0)
            if "train_misclass" in dfs[0]:
                misclass = result * 100
            else:
                rmse = result

            if "test_auc" in dfs[0]:
                auc = np.mean([df["test_auc"].iloc[-1] for df in dfs],
                              axis=0)
            else:
                auc = None

            row = {
                "auc": auc,
                "model": model,
                "task": decoding_task,
                "subset": decoding_type,
                "misclass": misclass,
                "rmse": rmse,
                "n": len(dfs),
                "n_epochs": len(dfs[0])-1
            }
            result_df = result_df.append(row, ignore_index=True)
    return result_df


# taken from:
# donghao.org/2018/04/10/the-problem-of-bool-type-in-argparse-of-python-2-7/
def str2bool(value):
    return value.lower() == 'true'


def create_parser_with_args(arg_type_map):
    parser = argparse.ArgumentParser()
    for arg, arg_type in arg_type_map:
        parser.add_argument("--" + arg, type=arg_type)
    return parser


def parse_args(arg_type_map):
    parser = create_parser_with_args(arg_type_map)
    known, unknown = parser.parse_known_args()
    if unknown:
        logging.error("I don't know these arguments:\n{}".format(unknown))
        exit()
    return vars(known)


def parse_run_args():
    arg_type_map = [
        ['batch_size', int],
        ['cuda', str2bool],
        ['eval_folder', str],
        ['final_conv_length', int],
        ['gradient_clip', float],
        ['init_lr', float],
        ['input_time_length', int],
        ['lazy_loading', str2bool],
        ['l2_decay', float],
        ['max_epochs', int],
        ['model_constraint', str],
        ['model_name', str],
        ['n_chan_factor', int],
        ['n_chans', int],
        ['n_folds', int],
        ['n_recordings', int],
        ['n_start_chans', int],
        ['num_workers', int],
        ['result_folder', str],
        ['run_on_abnormals', str2bool],
        ['run_on_normals', str2bool],
        ['seed', int],
        ['shuffle_folds', str2bool],
        ['stride_before_pool', str2bool],
        ['task', str],
        ['train_folder', str],
        ['weight_decay', float]
    ]
    # here, Nones are ok
    return parse_args(arg_type_map)


def parse_submit_args():
    arg_type_map = [
        ['configs_file', str],
        ['python_file', str],
        ['queue', str],
        ['conda_env_name', str],
        ['python_path', str],
        ['start', int],
        ['stop', int]
    ]
    args = parse_args(arg_type_map)
    # all the args have to be given, no Nones allowed
    # assert None not in args.values(), "arg unset {}".format(args)
    return args
