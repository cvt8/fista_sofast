import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import dill # type: ignore
from grad import grad_f, prox_g, penalty_g
from visualisation import plot_sparsity_evolution, plot_non_zero_probabilities


def has_converged(prev_value, curr_value, tol=1e-2):
    """
    Check if the algorithm has converged based on the change in objective values.
    """
    return np.abs(curr_value - prev_value) < tol


#The papers algorithms
# Algo1: P-PG
def algo1(theta_init, Y, max_iter=2000, gamma=0.1):
    theta = theta_init
    prev_objective = float("inf")
    for k in range(max_iter):
        num_samples = int(np.sqrt(k + 1))
        grad = grad_f(theta, Y, num_samples) 
        theta -= gamma * grad
        theta = prox_g(theta, gamma, lambda_reg, mu_reg)

        # Monitor convergence
        curr_objective = penalty_g(theta, lambda_reg, mu_reg) + np.sum(grad)
        if has_converged(prev_objective, curr_objective):
            print(f"Converged at iteration {k}")
            break
        prev_objective = curr_objective

    return theta


# Algo2: P-FISTA (t_n = O(n))
def algo2(theta_init, Y, max_iter=2000, gamma=0.1):
    theta = theta_init
    theta_old = np.copy(theta_init)
    t = 1
    prev_objective = float("inf")
    for k in range(max_iter):
        y = theta + (t - 1) / (t + 1) * (theta - theta_old)
        num_samples = min((k + 1) ** 3, 10000) # limit the number of samples
        grad = grad_f(y, Y, num_samples)
        theta_new = prox_g(y - gamma * grad, gamma, lambda_reg, mu_reg)
        theta_old = np.copy(theta)
        theta = np.copy(theta_new)
        t += 1

        # Monitor convergence
        curr_objective = penalty_g(theta, lambda_reg, mu_reg) + np.sum(grad)
        if has_converged(prev_objective, curr_objective):
            print(f"Converged at iteration {k}")
            break
        prev_objective = curr_objective
    return theta


# Algo3: P-FISTA (t_n = O(sqrt(n)))
def algo3(theta_init, Y, max_iter=2000, gamma=0.1):
    theta = theta_init
    theta_old = np.copy(theta_init)
    t = 1
    prev_objective = float("inf")
    for k in range(max_iter):
        y = theta + (t - 1) / (t + 1) * (theta - theta_old)
        num_samples = min((k + 1) ** 3, 10000) # limit the number of samples
        grad = grad_f(y, Y, num_samples)
        theta_new = prox_g(y - gamma * grad, gamma, lambda_reg, mu_reg)
        theta_old = np.copy(theta)
        theta = np.copy(theta_new)
        t = np.sqrt(k+1)

        # Monitor convergence
        curr_objective = penalty_g(theta, lambda_reg, mu_reg) + np.sum(grad)
        if has_converged(prev_objective, curr_objective):
            print(f"Converged at iteration {k}")
            break
        prev_objective = curr_objective
    return theta


# Algo4: P-FISTA (t_n = O(n^epsilon))
def algo4(theta_init, Y, max_iter=2000, gamma=0.1, epsilon=0.1):
    theta = theta_init
    theta_old = np.copy(theta_init)
    t = 1
    prev_objective = float("inf")
    for k in range(max_iter):
        y = theta + (t - 1) / (t + 1) * (theta - theta_old)
        num_samples = min((k + 1) ** 3, 10000) # limit the number of samples
        grad = grad_f(y, Y, num_samples)
        theta_new = prox_g(y - gamma * grad, gamma, lambda_reg, mu_reg)
        theta_old = np.copy(theta)
        theta = np.copy(theta_new)
        t = np.power(k+1, epsilon)

        # Monitor convergence
        curr_objective = penalty_g(theta, lambda_reg, mu_reg) + np.sum(grad)
        if has_converged(prev_objective, curr_objective):
            print(f"Converged at iteration {k}")
            break
        prev_objective = curr_objective
    return theta


# Algo5: P-PG with Accumulated Gradient
def algo5(theta_init, Y, max_iter=2000, gamma=0.1):
    theta = theta_init
    grad_accumulated = np.zeros_like(theta)
    prev_objective = float("inf")
    for k in range(max_iter):
        num_samples = 10  # A few samples at each iteration
        grad = grad_f(theta, Y, num_samples)
        grad_accumulated = (1 - 1 / (k + 1)) * grad_accumulated + (1 / (k + 1)) * grad
        theta -= gamma * grad_accumulated
        theta = prox_g(theta, gamma, lambda_reg, mu_reg)

        # Monitor convergence
        curr_objective = penalty_g(theta, lambda_reg, mu_reg) + np.sum(grad)
        if has_converged(prev_objective, curr_objective):
            print(f"Converged at iteration {k}")
            break
        prev_objective = curr_objective
    return theta


# Run the five algorithms and compare their performance
def run_algorithms(Y, p, max_iter=2000, runs=10):
    iterations_to_track = [50, 500, 1000, 1500, 2000]

    results_sparsity = {'Algo1': [], 'Algo2': [], 'Algo3': [], 'Algo4': [], 'Algo5': []}
    non_zero_probabilities = {'Algo1': [], 'Algo2': [], 'Algo3': [], 'Algo4': [], 'Algo5': []}

    for _ in range(runs):
        print(f"Run: {_ + 1}/{runs}")
        theta_init = np.random.normal(0, 1, size=(p, p))

        # Run each algorithm
        for algo_name, algo_func in zip(
            ['Algo1', 'Algo2', 'Algo3', 'Algo4', 'Algo5'],
            [algo1, algo2, algo3, algo4, algo5]
        ):
            print(f"Running {algo_name}")
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
    results_sparsity, non_zero_probabilities, iterations_to_track = run_algorithms(Y, p, max_iter=2000, runs=10)

    # Sauvegarde des résultats
    with open(f'results_data/fista_use_paper_results_{p}.pkl', 'wb') as f:
        dill.dump((results_sparsity, non_zero_probabilities, iterations_to_track), f)

    # Plot sparsity evolution
    plot_sparsity_evolution(results_sparsity, iterations_to_track)

    # Plot non-zero probabilities heatmaps
    plot_non_zero_probabilities(non_zero_probabilities)