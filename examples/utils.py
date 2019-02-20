import pandas as pd
import numpy as np
import argparse
import os
import re


def dfs_from_dir(path):
    df = pd.DataFrame()
    i = 0
    while True:
        df_file = path+"epochs_df_"+str(i)+".csv"
        if not os.path.exists(df_file):
            return df
        df = df.append(pd.read_csv(df_file).iloc[-1], ignore_index=True)
        i += 1


def df_list_from_dir(path):
    df = []
    i = 0
    while True:
        df_file = path+"epochs_df_"+str(i)+".csv"
        if not os.path.exists(df_file):
            return df
        df.append(pd.read_csv(df_file))
        i += 1


def read_network_results_without_resampy(directory, models, decoding_tasks):
    result_df = pd.DataFrame()
    for model in models:
        for decoding_task in decoding_tasks:
            for decoding_type in ["cv", "eval"]:
                curr_path = os.path.join(directory, model, decoding_task, decoding_type, "")
                if os.path.exists(curr_path):
                    # settings = os.listdir(curr_path)
                    # for setting in settings:
                    #     path = os.path.join(curr_path, setting, "")
                    dfs = df_list_from_dir(curr_path)
                    rmse = np.nan
                    misclass = np.nan
                    misclass_or_rmse = "misclass" if "train_misclass" in dfs[0] else "rmse"
                    result = np.mean([df["test_"+misclass_or_rmse].iloc[-1] for df in dfs], axis=0)
                    if "train_misclass" in dfs[0]:
                        misclass = result * 100
                    else:
                        rmse = result

                    if "test_auc" in dfs[0]:
                        auc = np.mean([df["test_auc"].iloc[-1] for df in dfs], axis=0)
                    else:
                        auc = None

                    # resampy_version = re.findall("resampy([0-9]+?.[0-9]+?.[0-9]+?)", setting)
                    # if len(resampy_version) == 0:
                    #     resampy_version = [np.nan]
                    row = {
                        "auc": auc,
                        "model": model,
                        # "rejecting": not "no_rejecting" in setting,
                        "task": decoding_task,
                        "subset": decoding_type,
                        # "clipping": "after" if "after" in setting else "before",
                        # "resampy": resampy_version[0],
                        "misclass": misclass,
                        "rmse": rmse,
                        "n": len(dfs),
                        "n_epochs": len(dfs[0])-1
                    }
                    result_df = result_df.append(row, ignore_index=True)
    return result_df


def read_network_results(directory, models, decoding_tasks):
    result_df = pd.DataFrame()
    for model in models:
        for decoding_task in decoding_tasks:
            for decoding_type in ["cv", "eval"]:
                curr_path = os.path.join(directory, model, decoding_task, decoding_type, "")
                if os.path.exists(curr_path):
                    settings = os.listdir(curr_path)
                    for setting in settings:
                        path = os.path.join(curr_path, setting, "")
                        dfs = df_list_from_dir(path)
                        rmse = np.nan
                        misclass = np.nan
                        misclass_or_rmse = "misclass" if "train_misclass" in dfs[0] else "rmse"
                        result = np.mean([df["test_"+misclass_or_rmse].iloc[-1] for df in dfs], axis=0)
                        if "train_misclass" in dfs[0]:
                            misclass = result * 100
                        else:
                            rmse = result

                        resampy_version = re.findall("resampy([0-9]+?.[0-9]+?.[0-9]+?)", setting)
                        if len(resampy_version) == 0:
                            resampy_version = [np.nan]
                        row = {
                            "model": model,
                            "rejecting": not "no_rejecting" in setting,
                            "task": decoding_task,
                            "subset": decoding_type,
                            "clipping": "after" if "after" in setting else "before",
                            "resampy": resampy_version[0],
                            "misclass": misclass,
                            "rmse": rmse,
                            "n": len(dfs),
                            "n_epochs": len(dfs[0])-1
                        }
                        result_df = result_df.append(row, ignore_index=True)
    return result_df


def parse_run_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument("--eval_folder", required=True, type=str)
    parser.add_argument("--final_conv_length", required=True, type=int)
    parser.add_argument("--init_lr", required=True, type=float)
    parser.add_argument("--input_time_length", required=True, type=int)
    parser.add_argument('--lazy_loading', dest='lazy_loading',
                        action='store_true')
    parser.add_argument('--no-lazy_loading', dest='lazy_loading',
                        action='store_false')
    parser.add_argument("--max_epochs", required=True, type=int)
    parser.add_argument("--model_constraint", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--n_chan_factor", required=True, type=str)  # parse yourself
    parser.add_argument("--n_chans", required=True, type=int)
    parser.add_argument("--n_folds", required=True, type=int)
    parser.add_argument("--n_recordings", required=True, type=str)  # parse yourself
    parser.add_argument("--n_start_chans", required=True, type=int)
    parser.add_argument("--num_workers", required=True, type=int)
    parser.add_argument("--result_folder", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument('--shuffle_folds', dest='shuffle_folds',
                        action='store_true')
    parser.add_argument('--no-shuffle_folds', dest='shuffle_folds',
                        action='store_false')
    parser.add_argument('--stride_before_pool', dest='stride_before_pool',
                        action='store_true')
    parser.add_argument('--no-stride_before_pool', dest='stride_before_pool',
                        action='store_false')
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--train_folder", required=True, type=str)
    parser.add_argument("--weight_decay", required=True, type=float)

    known, unknown = parser.parse_known_args()
    if unknown:
        print("I don't know these run arguments")
        print(unknown)
        exit()

    known_vars = vars(known)
    if known_vars["n_recordings"] in ["nan", "None"]:
        known_vars["n_recordings"] = None
    else:  # assume integer
        known_vars["n_recordings"] = int(known_vars["n_recordings"])

    if known_vars["n_chan_factor"] in ["nan", "None"]:
        known_vars["n_chan_factor"] = None
    else:  # assume integer
        known_vars["n_chan_factor"] = int(known_vars["n_chan_factor"])

    if known_vars["model_constraint"] in ["nan", "None"]:
        known_vars["model_constraint"] = None

    if known_vars["result_folder"] in ["nan", "None"]:
        known_vars["result_folder"] = None

    if known_vars["eval_folder"] in ["nan", "None"]:
        known_vars["eval_folder"] = None

    return known_vars


def parse_submit_args():
    args = [
        ['configs_file', str],
        ['scipt_file', str],
        ['queue', str],
        ['n_parallel', int]
    ]

    parser = argparse.ArgumentParser()
    for arg, type in args:
        parser.add_argument("--" + arg, required=False, type=type)

    known, unknown = parser.parse_known_args()
    if unknown:
        print("I don't know these submit arguments")
        print(unknown)
        exit()

    known_vars = vars(known)
    return known_vars
