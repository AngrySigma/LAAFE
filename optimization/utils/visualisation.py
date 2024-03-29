import matplotlib.pyplot as plt
import numpy as np


def plot_metrics_history(
    metrics_history: list[float], plot_path: str | None = None, metric="Accuracy"
):
    # maybe, it would be better to get all the metrics from the optimizer
    # and plot populations
    plt.close()
    plt.plot(metrics_history, alpha=0.5, color="blue")
    plt.plot(
        np.maximum.accumulate(metrics_history), alpha=0.75, color="red", linewidth=2
    )
    plt.xlabel("Iteration")
    plt.ylabel(metric)
    plt.legend(["Accuracy", "Best Accuracy"])
    if plot_path is not None:
        plt.savefig(plot_path)
    else:
        plt.show()
