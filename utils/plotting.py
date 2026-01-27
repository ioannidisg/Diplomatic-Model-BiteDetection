import matplotlib.pyplot as plt

def plot_bite_intervals(detected_intervals, ground_truth_intervals, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 3))

    # Ground truth (κάτω)
    for (start, end) in ground_truth_intervals:
        ax.hlines(y=0, xmin=start, xmax=end, colors='green', linewidth=4, label='Ground Truth')

    # Detected (πάνω)
    for (start, end) in detected_intervals:
        ax.hlines(y=1, xmin=start, xmax=end, colors='red', linewidth=4, label='Detected')

    ax.set_xlabel("Time")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Ground Truth", "Detected"])
    ax.set_title("Bite Detection vs Ground Truth")
    ax.grid(True)

    # Avoid duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
