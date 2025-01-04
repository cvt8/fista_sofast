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
    print("log_likelihood", f_theta)
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

def grad_f(theta, Y, num_samples=100):
    """
    Approximates the gradient of f(θ) using Monte Carlo sampling.
    """
    # Use Wolff sampling or another appropriate MCMC method
    samples = wolff_sampler(theta, num_samples)
    
    # Compute Monte Carlo sums for expectations
    Y_mean = np.mean(Y, axis=0)
    interaction = np.mean(np.einsum('ij,ik->ijk', Y, Y), axis=0)
    grad_theta = interaction - theta @ Y_mean  # Adjust gradient as needed
    print("grad_f", grad_theta)
    return grad_theta



def p_pg(theta_init, Y, max_iter=2000, gamma=0.1):
    """
    Implements the P-PG algorithm.
    """
    theta = theta_init
    for k in range(max_iter):
        grad = grad_f(theta, Y, num_samples=int(np.sqrt(k + 1)))  # Adjust samples
        theta -= gamma * grad
        theta = prox_g(theta, gamma)
    print("p_pg", theta)
    return theta

def p_fista(theta_init, Y, max_iter=2000, gamma=0.1):
    """
    Implements the P-FISTA algorithm.
    """
    theta = theta_init
    theta_old = np.copy(theta_init)
    t = 1
    for k in range(max_iter):
        y = theta + (t - 1) / (t + 1) * (theta - theta_old)
        grad = grad_f(y, Y, num_samples=k**3)  # Larger Monte Carlo samples
        theta_new = prox_g(y - gamma * grad, gamma)
        theta_old = np.copy(theta)
        theta = np.copy(theta_new)
        t += 1
    print("p_fista", theta)
    return theta

# Visualize sparsity
def plot_sparsity_results(results, iterations):
    plt.figure(figsize=(10, 6))
    for name, sparsities in results.items():
        plt.plot(iterations, sparsities, label=name)
    plt.xlabel('Iterations')
    plt.ylabel('Number of Non-Zero Components')
    plt.legend()
    plt.title('Sparsity Comparison Across Algorithms')
    plt.show()

# Run multiple algorithms and compare their performance
def run_algorithms():
    max_iter = 2000
    runs = 100  # Number of independent runs for probabilistic results
    iterations_to_track = [50, 500, 1000, 1500, 2000]

    results_sparsity = {'P-PG': [], 'P-FISTA': []}
    non_zero_probabilities = {'P-PG': [], 'P-FISTA': []}

    for _ in range(runs):
        print(f"Run: {_ + 1}")
        # Initialize theta
        theta_init = np.random.normal(0, 1, size=(p, p))
        
        # Run P-PG
        theta_p_pg = p_pg(np.copy(theta_init), Y, max_iter=max_iter)
        sparsity_p_pg = [np.count_nonzero(theta_p_pg[:i]) for i in iterations_to_track]
        results_sparsity['P-PG'].append(sparsity_p_pg)
        non_zero_probabilities['P-PG'].append(theta_p_pg != 0)
        
        # Run P-FISTA
        theta_p_fista = p_fista(np.copy(theta_init), Y, max_iter=max_iter)
        sparsity_p_fista = [np.count_nonzero(theta_p_fista[:i]) for i in iterations_to_track]
        results_sparsity['P-FISTA'].append(sparsity_p_fista)
        non_zero_probabilities['P-FISTA'].append(theta_p_fista != 0)

    # Aggregate non-zero probabilities
    for key in non_zero_probabilities:
        non_zero_probabilities[key] = np.mean(non_zero_probabilities[key], axis=0)

    return results_sparsity, non_zero_probabilities, iterations_to_track

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


    # Test du proximal operator
    gamma_test = 0.1  # Valeur de test pour le pas
    theta_test = np.copy(theta_true)  # Exemple d'initialisation
    theta_prox = prox_g(theta_test, gamma_test)
    print(np.linalg.norm(theta_prox - theta_test, ord=2))  # Vérification de la proximité


    # Run simulations and collect results
    results_sparsity, non_zero_probabilities, iterations_to_track = run_algorithms()
    dill.dump_session('fista_use_paper.db')

    # Plot sparsity evolution
    plot_sparsity_evolution(results_sparsity, iterations_to_track)

    # Plot non-zero probabilities heatmaps
    plot_non_zero_probabilities(non_zero_probabilities)
