import numpy as np
import matplotlib.pyplot as plt

# Visualisation des résultats

def plot_sparsity_results(results, iterations):
    plt.figure(figsize=(10, 6))
    for name, sparsities in results.items():
        plt.plot(iterations, sparsities, label=name)
    plt.xlabel('Iterations')
    plt.ylabel('Number of Non-Zero Components')
    plt.legend()
    plt.title('Sparsity Comparison Across Algorithms')
    plt.show()

def plot_sparsity_evolution(results_sparsity, iterations_to_track):
    plt.figure(figsize=(10, 6))
    for algo, sparsity_values in results_sparsity.items():
        mean_sparsity = np.mean(sparsity_values, axis=0)
        plt.plot(iterations_to_track, mean_sparsity, label=algo)
    plt.xlabel("Iterations")
    plt.ylabel("Number of Non-Zero Components")
    plt.title("Sparsity Evolution Across Iterations")
    plt.legend()
    plt.savefig(f"sparsity_evolution_{p}.png")
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
    plt.savefig(f"non_zero_probabilities_{p}.png")



if __name__ == "__main__":
    # Test des fonctions dans un cas simple
    p = 2
    results_sparsity = {'Algo1': [[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], 
                                  [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], 
                                  [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]], 
                        'Algo2': [[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], 
                                            [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]], 
                        'Algo3': [[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], 
                                  [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]], 
                        'Algo4': [[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], 
                                  [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]], 
                        'Algo5': [[3, 3, 3, 3, 3], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], 
                                  [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]]}
    non_zero_probabilities = {'Algo1': np.array([[1., 1.], [1., 1.]]), 'Algo2': np.array([[1., 1.],
       [1., 1.]]), 'Algo3': np.array([[1., 1.],
       [1., 1.]]), 'Algo4': np.array([[1., 1.],
       [1., 1.]]), 'Algo5': np.array([[1. , 0.9],
       [0.9, 1. ]])}
    iterations_to_track = [50, 500, 1000, 1500, 2000]
                                              
    plot_sparsity_evolution(results_sparsity, iterations_to_track)
    plot_non_zero_probabilities(non_zero_probabilities)