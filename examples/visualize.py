import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
import pandas as pd
import numpy as np
import pylab
import os
sns.set(color_codes=True)

from .utils import df_list_from_dir


def plot_runtime(dfs):
    plt.figure(figsize=(12, 5))
    for df_i, df in enumerate(dfs):
        runtime = df["runtime"][1:] / 60
        runtime.plot(label="fold {} ({:.2f} min/epoch in average)"
                     .format(df_i, np.mean(runtime)))
    plt.xlabel("epoch")
    plt.ylabel("time [min]")
    plt.legend()


def plot_learning_with_two_scales(df, ylim, just_test=False, out_dir=None):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    misclass_or_rmse = "misclass" if "train_misclass" in df[0] else "rmse"
    if misclass_or_rmse == "rmse":
        factor = 1
        second_y_label = misclass_or_rmse + " [years]"
        second_y_lim = 0, ylim
    else:
        factor = 100
        second_y_label = misclass_or_rmse + " [%]"
        second_y_lim = 0, ylim

    if not just_test:
        for i in range(len(df)):
            ax1.plot(df[i].train_loss, color="g", linewidth=.5, label="")
    mean_loss = np.mean([df[i].train_loss for i in range(len(df))], axis=0)
    mean_loss_final = np.mean([df[i].train_loss.iloc[-1] for i in range(len(df))],
                              axis=0)
    std_loss_final = np.std([df[i].train_loss.iloc[-1] for i in range(len(df))],
                            axis=0)
    l1 = ax1.plot(mean_loss, color="g",
                  label="train_loss ({:.2f} $\pm$ {:.2f})".format(mean_loss_final,
                                                              std_loss_final))

    for i in range(len(df)):
        ax1.plot(df[i].test_loss,  color="r", linewidth=.5, label="")
    mean_loss = np.mean([df[i].test_loss for i in range(len(df))], axis=0)
    mean_loss_final = np.mean([df[i].test_loss.iloc[-1] for i in range(len(df))],
                              axis=0)
    std_loss_final = np.std([df[i].test_loss.iloc[-1] for i in range(len(df))],
                            axis=0)
    l2 = ax1.plot(mean_loss, color="r",
                  label="test_loss ({:.2f} $\pm$ {:.2f})".format(mean_loss_final,
                                                             std_loss_final))

    ax1.set_xlabel('epoch')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('loss', color='r')
    ax1.tick_params('y', colors='r')
    ax1.set_yscale("symlog")
    ax1.set_ylim(0, 2)
    y_ticks = [0, 1, 2]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels([str(i) for i in y_ticks])

    # plot on second y axis
    ax2 = ax1.twinx()

    if not just_test:
        for i in range(len(df)):
            ax2.plot(df[i]["train_" + misclass_or_rmse] * factor,
                     linestyle="--", color="b", linewidth=.5, label="")
    mean_misclass = np.mean([df[i]["train_" + misclass_or_rmse] * factor
                             for i in range(len(df))], axis=0)
    mean_misclass_final = np.mean([df[i]["train_" + misclass_or_rmse].iloc[-1]
                                   * factor for i in range(len(df))],
                                  axis=0)
    std_misclass_final = np.std([df[i]["train_" + misclass_or_rmse].iloc[-1]
                                 * factor for i in range(len(df))], axis=0)
    l3 = ax2.plot(mean_misclass, linestyle="--", color="b",
                  label="train_" + misclass_or_rmse + " ({:.2f} $\pm$ {:.2f})"
                  .format(mean_misclass_final, std_misclass_final))

    for i in range(len(df)):
        ax2.plot(df[i]["test_" + misclass_or_rmse] * factor, linestyle="--",
                 color="orange", linewidth=.5, label="")
    mean_misclass = np.mean([df[i]["test_" + misclass_or_rmse] * factor
                             for i in range(len(df))], axis=0)
    mean_misclass_final = np.mean([df[i]["test_" + misclass_or_rmse].iloc[-1]
                                   * factor for i in range(len(df))], axis=0)
    std_misclass_final = np.std([df[i]["test_" + misclass_or_rmse].iloc[-1]
                                 * factor for i in range(len(df))], axis=0)
    l4 = ax2.plot(mean_misclass, linestyle="--", color="orange",
                  label="test_" + misclass_or_rmse + " ({:.2f} $\pm$ {:.2f})"
                  .format(mean_misclass_final, std_misclass_final))

    ax2.set_ylabel(second_y_label, color='orange')
    ax2.tick_params('y', colors='orange')
    ax2.set_ylim(second_y_lim)

    ls = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in ls]
    plt.legend(ls, labels, ncol=2, loc="best")

    fig.tight_layout()
    plt.xlim(0, len(df[0]))
    if out_dir is not None:
        plt.savefig(out_dir+"learning.pdf", bbox_inches="tight")


def plot_result_overview(result_directory, decoding_tasks, models, metric_name,
                         fs=20):
    result_df = pd.DataFrame()
    splits = ["cv", "eval"]
    i = 0
    for model, decoding_task, decoding_type in product(
            models, decoding_tasks, splits):
        path = os.path.join(result_directory, model, decoding_task,
                            decoding_type) + "/"
        if os.path.exists(path):
            dfs = df_list_from_dir(path)
            assert len(dfs) == 5, (
                "expected 5 dataframes from 5 folds/repetitions")
            if "test_misclass" in dfs[0]:
                misclass_or_rmse = "misclass"
                factor = 100
            else:
                misclass_or_rmse = "rmse"
                factor = 1
            misclasses = [d["test_"+misclass_or_rmse].iloc[-1]
                          for d in dfs]
            for misclass in misclasses:
                result_df = result_df.append({
                    "model": model,
                    "subset": decoding_type,
                    "task": decoding_task,
                    "misclass/rmse": misclass*factor},
                    ignore_index=True)
            i += 1

    n_colors = len(decoding_tasks)*len(models)
    # cm = pylab.get_cmap('tab10')
    # cm = pylab.get_cmap('viridis')
    cm = pylab.get_cmap('tab20')
    random_width = .3
    fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True,
                             figsize=(18, 10))
    for j, decoding_type in enumerate(splits):
        xticklabels = []
        ax = axes[j]
        ax.set_title(decoding_type, fontsize=fs)
        for i, (decoding_task, model) in enumerate(product(
                decoding_tasks, models)):
            d = result_df[(result_df.model == model) &
                          (result_df.task == decoding_task) &
                          (result_df.subset == decoding_type)]
            ax.scatter(np.random.rand(len(d))*random_width+i+.5+random_width,
                       d["misclass/rmse"],
                       color=cm(i/n_colors), label="")
            ax.scatter(np.array([i+1]),
                       np.array([d["misclass/rmse"].mean()]),
                       marker="^", color=cm(1.*i/n_colors),
                       s=200, facecolor="none",
                       label="{:.2f} ($\pm$ {:.2f})"
                       .format(d["misclass/rmse"].mean(),
                               d["misclass/rmse"].std()))
            xticklabels.append('\n'.join([model, decoding_task]))

        ylabel = metric_name + " [%]" + " / RMSE [years]"
        ax.set_ylabel(ylabel, fontsize=fs)
        ax.tick_params(axis="y", labelsize=fs)
        ax.set_xlim(0, (len(xticklabels) + 1))
        ax.set_ylim(5, 55)
        ax.legend(fontsize=fs-10, loc="upper center",
                  ncol=int(len(models)*len(decoding_tasks)/2))

    plt.xticks(np.arange(1, len(xticklabels) + 2),
               xticklabels, rotation=90, fontsize=fs)
    plt.xlabel("experiment", fontsize=fs)

