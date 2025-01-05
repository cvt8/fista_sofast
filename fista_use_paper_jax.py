import jax
import jax.numpy as jnp
from jax import jit
import concurrent.futures
import dill
import matplotlib.pyplot as plt
import numpy as np
from grad import grad_f, prox_g, penalty_g
from visualisation import plot_sparsity_evolution, plot_non_zero_probabilities

@jit
def has_converged(prev_value, curr_value, tol=1e-2):
    """
    Check if the algorithm has converged based on the change in objective values.
    """
    return jnp.abs(curr_value - prev_value) < tol


#The papers algorithms
# Algo1: P-PG
@jit
def algo1(theta_init, Y, max_iter=2000, gamma=0.1):
    theta = theta_init
    prev_objective = float("inf")
    for k in range(max_iter):
        num_samples = int(jnp.sqrt(k + 1))
        grad = grad_f(theta, Y, num_samples) 
        theta -= gamma * grad
        theta = prox_g(theta, gamma, lambda_reg, mu_reg)

        # Monitor convergence
        curr_objective = penalty_g(theta, lambda_reg, mu_reg) + jnp.sum(grad)
        if has_converged(prev_objective, curr_objective):
            print(f"Converged at iteration {k}")
            break
        prev_objective = curr_objective

    return theta


# Algo2: P-FISTA (t_n = O(n))
@jit
def algo2(theta_init, Y, max_iter=2000, gamma=0.1):
    theta = theta_init
    theta_old = jnp.copy(theta_init)
    t = 1
    prev_objective = float("inf")
    for k in range(max_iter):
        y = theta + (t - 1) / (t + 1) * (theta - theta_old)
        num_samples = min((k + 1) ** 3, 10000) # limit the number of samples
        grad = grad_f(y, Y, num_samples)
        theta_new = prox_g(y - gamma * grad, gamma, lambda_reg, mu_reg)
        theta_old = jnp.copy(theta)
        theta = jnp.copy(theta_new)
        t += 1

        # Monitor convergence
        curr_objective = penalty_g(theta, lambda_reg, mu_reg) + jnp.sum(grad)
        if has_converged(prev_objective, curr_objective):
            print(f"Converged at iteration {k}")
            break
        prev_objective = curr_objective
    return theta


# Algo3: P-FISTA (t_n = O(sqrt(n)))
@jit
def algo3(theta_init, Y, max_iter=2000, gamma=0.1):
    theta = theta_init
    theta_old = jnp.copy(theta_init)
    t = 1
    prev_objective = float("inf")
    for k in range(max_iter):
        y = theta + (t - 1) / (t + 1) * (theta - theta_old)
        num_samples = min((k + 1) ** 3, 10000) # limit the number of samples
        grad = grad_f(y, Y, num_samples)
        theta_new = prox_g(y - gamma * grad, gamma, lambda_reg, mu_reg)
        theta_old = jnp.copy(theta)
        theta = jnp.copy(theta_new)
        t = jnp.sqrt(k+1)

        # Monitor convergence
        curr_objective = penalty_g(theta, lambda_reg, mu_reg) + jnp.sum(grad)
        if has_converged(prev_objective, curr_objective):
            print(f"Converged at iteration {k}")
            break
        prev_objective = curr_objective
    return theta


# Algo4: P-FISTA (t_n = O(n^epsilon))
@jit
def algo4(theta_init, Y, max_iter=2000, gamma=0.1, epsilon=0.1):
    theta = theta_init
    theta_old = jnp.copy(theta_init)
    t = 1
    prev_objective = float("inf")
    for k in range(max_iter):
        y = theta + (t - 1) / (t + 1) * (theta - theta_old)
        num_samples = min((k + 1) ** 3, 10000) # limit the number of samples
        grad = grad_f(y, Y, num_samples)
        theta_new = prox_g(y - gamma * grad, gamma, lambda_reg, mu_reg)
        theta_old = jnp.copy(theta)
        theta = jnp.copy(theta_new)
        t = jnp.power(k+1, epsilon)

        # Monitor convergence
        curr_objective = penalty_g(theta, lambda_reg, mu_reg) + jnp.sum(grad)
        if has_converged(prev_objective, curr_objective):
            print(f"Converged at iteration {k}")
            break
        prev_objective = curr_objective
    return theta


# Algo5: P-PG with Accumulated Gradient
@jit
def algo5(theta_init, Y, max_iter=2000, gamma=0.1):
    theta = theta_init
    grad_accumulated = jnp.zeros_like(theta)
    prev_objective = float("inf")
    for k in range(max_iter):
        num_samples = 10  # A few samples at each iteration
        grad = grad_f(theta, Y, num_samples)
        grad_accumulated = (1 - 1 / (k + 1)) * grad_accumulated + (1 / (k + 1)) * grad
        theta -= gamma * grad_accumulated
        theta = prox_g(theta, gamma, lambda_reg, mu_reg)

        # Monitor convergence
        curr_objective = penalty_g(theta, lambda_reg, mu_reg) + jnp.sum(grad)
        if has_converged(prev_objective, curr_objective):
            print(f"Converged at iteration {k}")
            break
        prev_objective = curr_objective
    return theta


# Run a single simulation for one run
def run_single_run(i, Y, p, max_iter):
    key = jax.random.PRNGKey(i)
    theta_init = jax.random.normal(key, (p, p))
    run_results_sparsity = {}
    run_non_zero_probabilities = {}

    for algo_name, algo_func in zip(
        ['Algo1', 'Algo2', 'Algo3', 'Algo4', 'Algo5'],
        [algo1, algo2, algo3, algo4, algo5]
    ):
        print(f"Run {i + 1}: Running {algo_name}")
        theta = algo_func(jnp.copy(theta_init), Y, max_iter=max_iter)
        sparsity = [jnp.count_nonzero(theta) for _ in iterations_to_track]
        run_results_sparsity[algo_name] = sparsity
        run_non_zero_probabilities[algo_name] = theta != 0

    return run_results_sparsity, run_non_zero_probabilities

# Parallelized runs
def run_algorithms(Y, p, max_iter=2000, runs=10):
    iterations_to_track = [50, 500, 1000, 1500, 2000]

    results_sparsity = {'Algo1': [], 'Algo2': [], 'Algo3': [], 'Algo4': [], 'Algo5': []}
    non_zero_probabilities = {'Algo1': [], 'Algo2': [], 'Algo3': [], 'Algo4': [], 'Algo5': []}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_single_run, i, Y, p, max_iter)
            for i in range(runs)
        ]
        for future in concurrent.futures.as_completed(futures):
            run_results_sparsity, run_non_zero_probabilities = future.result()
            for algo_name in ['Algo1', 'Algo2', 'Algo3', 'Algo4', 'Algo5']:
                results_sparsity[algo_name].append(run_results_sparsity[algo_name])
                non_zero_probabilities[algo_name].append(run_non_zero_probabilities[algo_name])

    # Aggregate non-zero probabilities
    for key in non_zero_probabilities:
        non_zero_probabilities[key] = jnp.mean(jnp.stack(non_zero_probabilities[key]), axis=0)

    return results_sparsity, non_zero_probabilities, iterations_to_track




if __name__ == "__main__":
    # Paramètres du modèle
    # Dimensions du problème
    jax.config.update("jax_enable_x64", True)  # Enable double precision
    N = 250  # Nombre d'observations
    p = 100  # Dimension du modèle (p x p)

     # Data generation
    np.random.seed(42)
    theta_true = np.zeros((p, p))
    upper_indices = np.triu_indices(p, k=1)
    num_upper = len(upper_indices[0])
    theta_true[upper_indices] = np.random.choice([0, 1], size=num_upper, p=[0.98, 0.02])
    np.fill_diagonal(theta_true, np.random.uniform(0.5, 1.0, size=p))
    Y = jnp.array(np.random.binomial(1, 0.5, size=(N, p)))


    # Définition de la pénalité g(θ)
    lambda_reg = 0.5 * jnp.sqrt(jnp.log(p) / N)  # Paramètre de régularisation L1
    mu_reg = 0.5  # Paramètre de régularisation L2

    # Simulations
    results_sparsity, non_zero_probabilities, iterations_to_track = run_algorithms(Y, p, max_iter=2000, runs=10)

    # Sauvegarde des résultats
    with open(f'fista_use_paper_results_{p}.pkl', 'wb') as f:
        dill.dump((results_sparsity, non_zero_probabilities, iterations_to_track), f)

    # Plot sparsity evolution
    plot_sparsity_evolution(results_sparsity, iterations_to_track)

    # Plot non-zero probabilities heatmaps
    plot_non_zero_probabilities(non_zero_probabilities)