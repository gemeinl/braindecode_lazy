import matplotlib.pyplot as plt
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
                  label="train_loss ({:.2f} +-{:.2f})".format(mean_loss_final,
                                                              std_loss_final))

    for i in range(len(df)):
        ax1.plot(df[i].test_loss,  color="r", linewidth=.5, label="")
    mean_loss = np.mean([df[i].test_loss for i in range(len(df))], axis=0)
    mean_loss_final = np.mean([df[i].test_loss.iloc[-1] for i in range(len(df))],
                              axis=0)
    std_loss_final = np.std([df[i].test_loss.iloc[-1] for i in range(len(df))],
                            axis=0)
    l2 = ax1.plot(mean_loss, color="r",
                  label="test_loss ({:.2f} +-{:.2f})".format(mean_loss_final,
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
                  label="train_" + misclass_or_rmse + " ({:.2f} +-{:.2f})"
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
                  label="test_" + misclass_or_rmse + " ({:.2f} +-{:.2f})"
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
    # misclass_or_rmse = "rmse" if "age" in decoding_tasks else "misclass"
    misclass_or_rmse = metric_name
    factor = 1 if misclass_or_rmse == "rmse" else 100
    decoding_types = ["cv", "eval"]

    result_df = pd.DataFrame()
    i = 0
    for model in models:
        for decoding_task in decoding_tasks:
            for decoding_type in decoding_types:
                path = os.path.join(result_directory, model, decoding_task,
                                    decoding_type) + "/"
                if os.path.exists(path):
                    dfs = df_list_from_dir(path)[:5]
                    misclasses = [d["test_"+misclass_or_rmse].iloc[-1]
                                  for d in dfs]
                    for misclass in misclasses:
                        result_df = result_df.append({
                            "model": model,
                            "subset": decoding_type,
                            "task": decoding_task,
                            misclass_or_rmse: misclass*factor},
                            ignore_index=True)
                    i += 1

    n_colors = len(decoding_tasks)*len(decoding_types)*len(models)
    cm = pylab.get_cmap('tab10')

    plt.figure(figsize=(12, 5))
    i = 0
    labels = []
    for decoding_task in decoding_tasks:
        for model in models:
            for decoding_type in decoding_types:
                d = result_df[(result_df.model == model) &
                              (result_df.task == decoding_task) &
                              (result_df.subset == decoding_type)]
                if len(d) > 0:
                    sns.regplot(np.random.rand(len(d))+i+.5, d[misclass_or_rmse],
                                fit_reg=False, color=cm(1.*i/n_colors))
                    sns.regplot(np.array([i+1]),
                                np.array([d[misclass_or_rmse].mean()]),
                                marker="^", fit_reg=False,
                                scatter_kws={'s': 200}, color=cm(1.*i/n_colors),
                                label="{:.2f} ($\pm$ {:.2f})"
                                .format(d[misclass_or_rmse].mean(),
                                        d[misclass_or_rmse].std()))
                labels.append('\n'.join([model, decoding_task, decoding_type]))
                i += 1
    if misclass_or_rmse == "rmse":
        ylabel = "RMSE [years]"
    else:
        ylabel = metric_name + " [%]"
    plt.ylabel(ylabel, fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(np.arange(1, len(labels)+2), labels, rotation=90, fontsize=fs)
    #plt.xlabel("experiment", fontsize=fs)
    plt.legend(ncol=2, fontsize=fs)
    plt.xlim(.5, len(labels)+.5)
