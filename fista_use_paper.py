import numpy as np
import matplotlib.pyplot as plt
import dill


def penalty_g(theta):
    """
    Calcule la pénalité g(θ) = λ * ||θ_offdiag||_1 + μ * ||diag(θ)||_2^2.
    """
    diag_part = np.diag(theta)**2  # Partie quadratique (diagonale)
    off_diag_part = np.sum(np.abs(theta - np.diag(np.diag(theta))))  # Partie L1 hors diagonale
    res = lambda_reg * off_diag_part + mu_reg * np.sum(diag_part)
    print("penalty", res)
    return res

# Fonction log-vraisemblance pénalisée f(θ)
def log_likelihood_penalized(theta, Y):
    """
    Calcule -log(Z_θ) + termes de vraisemblance (θ'Y_mean et interaction binaire).
    """
    Y_mean = np.mean(Y, axis=0)
    interaction = np.mean(np.einsum('ij,ik->ijk', Y, Y), axis=0)
    
    # Log-partition (approximation via norme L2 pour éviter l'intractabilité)
    log_Z_theta = np.sum(np.log(1 + np.exp(theta @ theta.T)))

    # Calcul de -f(θ)
    f_theta = -log_Z_theta + np.sum(theta @ Y_mean.T) + np.sum(interaction * theta)
    return f_theta


def prox_g(theta, gamma):
    """
    Proximal operator for g(θ):
    L1-penalty on off-diagonal entries and L2 on diagonal entries.
    """
    diag = np.diag(theta)  # Extract diagonal
    off_diag = theta - np.diag(diag)  # Extract off-diagonal entries
    
    # Apply soft-thresholding for L1 penalty on off-diagonal
    prox_off_diag = np.sign(off_diag) * np.maximum(np.abs(off_diag) - gamma * lambda_reg, 0)
    
    # Apply shrinkage for L2 penalty on diagonal
    prox_diag = diag / (1 + gamma * mu_reg)
    
    # Reconstruct matrix
    prox_theta = prox_off_diag + np.diag(prox_diag)
    print("prox_g", prox_theta)
    return prox_theta

def wolff_sampler(theta, num_samples=100):
    """
    Wolff sampling algorithm for generating samples from a Boltzmann distribution.
    """
    samples = np.zeros((num_samples, theta.shape[0], theta.shape[1]))
    for i in range(num_samples):
        for j in range(theta.shape[0]):
            for k in range(j + 1, theta.shape[1]):
                theta_new = np.copy(theta)
                theta_new[j, k] = 1 - theta[j, k]  # Flip element
                prob_accept = np.exp(log_likelihood_penalized(theta_new, Y) - log_likelihood_penalized(theta, Y))
                if np.random.rand() < prob_accept:
                    theta = np.copy(theta_new)
        samples[i] = theta
    print("wolff_sampler", samples)
    return samples

def grad_f(theta, Y, num_samples):
    """
    Approximates the gradient of f(θ) using Wolff sampling.
    """
    # Use Wolff sampling to generate samples
    samples = wolff_sampler(theta, num_samples)
    grad = np.zeros_like(theta)
    for sample in samples:
        grad += sample - Y
    grad /= num_samples
    return grad


#The papers algorithm
# Algo1: P-PG
def algo1(theta_init, Y, max_iter=2000, gamma=0.1):
    theta = theta_init
    for k in range(max_iter):
        num_samples = int(np.sqrt(k + 1))
        grad = grad_f(theta, Y, num_samples)
        theta -= gamma * grad
        theta = prox_g(theta, gamma)
    return theta


# Algo2: P-FISTA (t_n = O(n))
def algo2(theta_init, Y, max_iter=2000, gamma=0.1):
    theta = theta_init
    theta_old = np.copy(theta_init)
    t = 1
    for k in range(max_iter):
        y = theta + (t - 1) / (t + 1) * (theta - theta_old)
        num_samples = (k + 1) ** 3
        grad = grad_f(y, Y, num_samples)
        theta_new = prox_g(y - gamma * grad, gamma)
        theta_old = np.copy(theta)
        theta = np.copy(theta_new)
        t += 1
    return theta


# Algo3: P-FISTA (t_n = O(sqrt(n)))
def algo3(theta_init, Y, max_iter=2000, gamma=0.1):
    theta = theta_init
    theta_old = np.copy(theta_init)
    t = 1
    for k in range(max_iter):
        y = theta + (t - 1) / (t + 1) * (theta - theta_old)
        num_samples = (k + 1) ** 3
        grad = grad_f(y, Y, num_samples)
        theta_new = prox_g(y - gamma * grad, gamma)
        theta_old = np.copy(theta)
        theta = np.copy(theta_new)
        t += 0.5  # Simulates O(sqrt(n)) growth
    return theta


# Algo4: P-FISTA (t_n = O(n^epsilon))
def algo4(theta_init, Y, max_iter=2000, gamma=0.1, epsilon=0.1):
    theta = theta_init
    theta_old = np.copy(theta_init)
    t = 1
    for k in range(max_iter):
        y = theta + (t - 1) / (t + 1) * (theta - theta_old)
        num_samples = (k + 1) ** 3
        grad = grad_f(y, Y, num_samples)
        theta_new = prox_g(y - gamma * grad, gamma)
        theta_old = np.copy(theta)
        theta = np.copy(theta_new)
        t += epsilon
    return theta


# Algo5: P-PG with Accumulated Gradient
def algo5(theta_init, Y, max_iter=2000, gamma=0.1):
    theta = theta_init
    grad_accumulated = np.zeros_like(theta)
    for k in range(max_iter):
        num_samples = 1  # Single sample at each iteration
        grad = grad_f(theta, Y, num_samples)
        grad_accumulated = (1 - 1 / (k + 1)) * grad_accumulated + (1 / (k + 1)) * grad
        theta -= gamma * grad_accumulated
        theta = prox_g(theta, gamma)
    return theta



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
    plt.savefig("sparsity_evolution.png")
    plt.close()


def plot_non_zero_probabilities(non_zero_probabilities):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for idx, (algo, probabilities) in enumerate(non_zero_probabilities.items()):
        ax = axes[idx]
        ax.imshow(probabilities, cmap="gray", aspect="auto")
        ax.set_title(f"Non-Zero Probability Heatmap: {algo}")
        ax.set_xlabel("Matrix Indices")
        ax.set_ylabel("Runs")
    plt.tight_layout()
    plt.savefig("non_zero_probabilities.png")
    plt.close()

# Run the five algorithms and compare their performance
def run_algorithms():
    max_iter = 2000
    runs = 10
    iterations_to_track = [50, 500, 1000, 1500, 2000]

    results_sparsity = {'Algo1': [], 'Algo2': [], 'Algo3': [], 'Algo4': [], 'Algo5': []}
    non_zero_probabilities = {'Algo1': [], 'Algo2': [], 'Algo3': [], 'Algo4': [], 'Algo5': []}

    for _ in range(runs):
        theta_init = np.random.normal(0, 1, size=(p, p))

        # Run each algorithm
        for algo_name, algo_func in zip(
            ['Algo1', 'Algo2', 'Algo3', 'Algo4', 'Algo5'],
            [algo1, algo2, algo3, algo4, algo5]
        ):
            theta = algo_func(np.copy(theta_init), Y, max_iter=max_iter)
            sparsity = [np.count_nonzero(theta) for _ in iterations_to_track]
            results_sparsity[algo_name].append(sparsity)
            non_zero_probabilities[algo_name].append(theta != 0)

    # Aggregate non-zero probabilities
    for key in non_zero_probabilities:
        non_zero_probabilities[key] = np.mean(non_zero_probabilities[key], axis=0)

    return results_sparsity, non_zero_probabilities, iterations_to_track


if __name__ == "__main__":
    # Paramètres du modèle
    # Dimensions du problème

    N = 250  # Nombre d'observations
    p = 100  # Dimension du modèle (p x p)
    np.random.seed(42)  # Répétabilité des résultats
    theta_true = np.zeros((p, p))
    upper_indices = np.triu_indices(p, k=1)  # Indices des éléments hors diagonale (supérieurs)
    num_upper = len(upper_indices[0])  # Nombre total d'éléments hors diagonale
    theta_true[upper_indices] = np.random.choice([0, 1], size=num_upper, p=[0.98, 0.02])
    np.fill_diagonal(theta_true, np.random.uniform(0.5, 1.0, size=p))  # Diagonale non nulle
    # Génération des données (modèle graphique binaire)
    Y = np.random.binomial(1, 0.5, size=(N, p))  # N échantillons de dimension p

    # Définition de la pénalité g(θ)
    lambda_reg = 0.5 * np.sqrt(np.log(p) / N)  # Paramètre de régularisation L1
    mu_reg = 0.5  # Paramètre de régularisation L2

    # Simulations
    results_sparsity, non_zero_probabilities, iterations_to_track = run_algorithms()
    dill.dump_session('fista_use_paper.db')

    # Plot sparsity evolution
    plot_sparsity_evolution(results_sparsity, iterations_to_track)

    # Plot non-zero probabilities heatmaps
    plot_non_zero_probabilities(non_zero_probabilities)
