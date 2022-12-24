import matplotlib.pyplot as plt
import numpy as np
import os

# Computes accuracy and average confidence for bin.
def get_uncalibrated_res(y_true, y_pred, confs_pred, M=15):
    '''
    :param y_true: (N,) ndarray, true label
    :param y_pred:(N,) ndarray, predicted label
    :param confs_pred:(N,) ndarray, prob的最大值，即confidence
    :param M:the length of the bins
    :return: (M,), ndarray, accuracy/average confidence/number of points in each bin
    '''
    bin_size = 1 / M # 每个bin的长度
    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)  # (1/M,2/M,...,M/M)
    accuracies, confidences, bin_lengths, bin_ratios = [], [], [], []
    N = len(y_true) # 样本数

    for conf_upper in upper_bounds:
        conf_lower = conf_upper - bin_size # 每个bin的下限
        filtered_tuples = [x for x in zip(y_pred, y_true, confs_pred)
            if x[2] > conf_lower and x[2] <= conf_upper] # 每个bin里面的点, 0是不被计入任何bin的
        if len(filtered_tuples) < 1: # 一个点也没有
            accuracy, avg_conf, len_bin, rat_bin = 0, 0, 0, 0
        else:
            correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels，计算预测label=真实label的点数
            len_bin = len(filtered_tuples)  # How many elements fall into given bin，点数即bin的长度
            avg_conf = (sum([x[2] for x in filtered_tuples]) / len_bin) # Avg confidence of BIN，bin中点的平均confi
            accuracy = float(correct) / len_bin # bin中点的正确率
            rat_bin = len_bin / N # 每个bin的样本占比

        accuracies.append(accuracy)
        confidences.append(avg_conf)
        bin_lengths.append(len_bin)
        bin_ratios.append(rat_bin)

    return np.array(accuracies), np.array(confidences), np.array(bin_lengths), np.array(bin_ratios)

def rel_diagram_sub(accs, confs, ece=None, data_name=None, model_name=None, name="Reliability Diagram", xname="Confidence", yname="Accuracy", path="./"):
    '''
    :param accs: (M,) ndarray, accuracy of each bin
    :param confs: (M,) ndarray, average confidence of each bin
    '''
    M = len(accs)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex="col", sharey="row")
    plt.plot(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1), linestyle="dashed", color="black") # 对角线图
    bin_size = 1 / M # 划分bin的宽度
    positions = np.arange(0 + bin_size / 2, 1 + bin_size / 2, bin_size)     # Center of each bin
    # Bars with outputs
    output_plt = ax.bar(positions, accs, width=bin_size, edgecolor="black", color="blue", zorder=0) # Plot gap first, so its below everything
    gap_plt = ax.bar(positions, confs - accs, bottom=accs, width=bin_size, edgecolor="red", hatch="/", color="red",
        alpha=0.3, linewidth=2, label="Gap", zorder=3)
    ax.text(0.55, 0.1, "ECE = {}%".format(round(ece * 100, 1)), size=14, backgroundcolor="grey")
    # Line plot with center line.
    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.legend(handles=[gap_plt, output_plt])
    ax.legend(loc=2, prop={"size": 14})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("{} on {}".format(model_name, data_name), fontsize=14)
    ax.set_xlabel(xname, fontsize=14, color="black")
    ax.set_ylabel(yname, fontsize=14, color="black")
    ax.xaxis.set_tick_params(labelsize=13)
    ax.yaxis.set_tick_params(labelsize=13)
    path =f"./Pictures/{data_name}/"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig("{}{}_{}.png".format(path, M, model_name))
