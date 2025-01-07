import numpy as np
import matplotlib.pyplot as plt
import dill

# Visualisation des résultats

def plot_sparsity_results(results, iterations):
    plt.figure(figsize=(10, 6))
    for name, sparsities in results.items():
        plt.plot(iterations, sparsities, label=name)
    plt.xlabel('Iterations')
    plt.ylabel('Number of Non-Zero Components')
    plt.legend()
    plt.title('Sparsity Comparison Across Algorithms')
    plt.savefig(f"figures/sparsity_comparison_{p}.png")
    plt.close()

def plot_sparsity_evolution(results_sparsity, iterations_to_track):
    plt.figure(figsize=(10, 6))
    for algo, sparsity_values in results_sparsity.items():
        mean_sparsity = np.mean(sparsity_values, axis=1)
        plt.plot(iterations_to_track, mean_sparsity, label=algo)
    plt.xlabel("Iterations")
    plt.ylabel("Number of Non-Zero Components")
    plt.title("Sparsity Evolution Across Iterations")
    plt.legend()
    plt.savefig(f"figures/sparsity_evolution_{p}.png")
    plt.close()

def plot_non_zero_components(results_sparsity, iterations):
    algo_names = list(results_sparsity.keys())
    non_zero_counts = np.array([results_sparsity[algo].mean(axis=1) for algo in algo_names])

    plt.figure(figsize=(12, 8))
    bar_width = 0.15
    indices = np.arange(len(algo_names))

    for i, iteration in enumerate(iterations):
        plt.bar(indices + i * bar_width, non_zero_counts[:, i], bar_width, label=f"Iteration {iteration}")

    plt.xlabel("Algorithms")
    plt.ylabel("Average Number of Non-Zero Components")
    plt.xticks(indices + bar_width * (len(iterations) - 1) / 2, algo_names)
    plt.legend()
    plt.title("Average Number of Non-Zero Components Across Algorithms and Iterations")
    plt.tight_layout()
    plt.savefig(f"figures/non_zero_components_{p}.png")
    plt.close()


def plot_non_zero_probabilities(non_zero_probabilities):
    axes = plt.subplots(1, len(non_zero_probabilities), figsize=(16, 6))[1]
    for idx, (algo, probabilities) in enumerate(non_zero_probabilities.items()):
        ax = axes[idx]
        ax.imshow(probabilities, cmap="gray", aspect="auto")
        ax.set_title(f"Non-Zero Probability Heatmap: \n {algo}")
        ax.set_xlabel("Matrix Indices")
        ax.set_ylabel("Runs")
    plt.tight_layout()
    plt.savefig(f"figures/non_zero_probabilities_{p}.png")
    plt.close()

def plot_zero_probabilities_heatmap(non_zero_probabilities, iterations):
    zero_probabilities = {algo: 1 - probs for algo, probs in non_zero_probabilities.items()}
    algo_names = list(zero_probabilities.keys())
    heatmap_data = np.array([zero_probabilities[algo].mean(axis=0)[:len(iterations)] for algo in algo_names])

    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_data, cmap="viridis", aspect="auto")
    plt.colorbar(label="Probability of Being Zero")
    plt.xticks(ticks=np.arange(len(iterations)), labels=iterations)
    plt.yticks(ticks=np.arange(len(algo_names)), labels=algo_names)
    plt.xlabel("Iterations")
    plt.ylabel("Algorithms")
    plt.title("Heatmap of Zero Probabilities Across Iterations and Algorithms")
    plt.tight_layout()
    plt.savefig(f"figures/zero_probabilities_heatmap_{p}.png")
    plt.close()


if __name__ == "__main__":
    p=10
    with open("results_data/fista_use_paper_results_10.pkl", "rb") as f:
        results_sparsity, non_zero_probabilities, iterations_to_track = dill.load(f)

    plot_sparsity_evolution(results_sparsity, iterations_to_track)
    plot_non_zero_components(results_sparsity, iterations_to_track)
    plot_non_zero_probabilities(non_zero_probabilities)
    plot_zero_probabilities_heatmap(non_zero_probabilities, iterations_to_track)
    plot_sparsity_results(results_sparsity, iterations_to_track)