import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import concurrent.futures

@jit
def penalty_g(theta, lambda_reg, mu_reg):
    """
    Calcule la pénalité g(θ) = λ * ||θ_offdiag||_1 + μ * ||diag(θ)||_2^2.
    """
    diag_part = jnp.diag(theta)**2  # Partie quadratique (diagonale)
    off_diag_part = jnp.sum(jnp.abs(theta - jnp.diag(jnp.diag(theta))))  # Partie L1 hors diagonale
    res = lambda_reg * off_diag_part + mu_reg * jnp.sum(diag_part)
    return res


# Fonction log-vraisemblance pénalisée f(θ)
@jit
def log_likelihood_penalized(theta, Y):
    """
    Calcule -log(Z_θ) + termes de vraisemblance (θ'Y_mean et interaction binaire).
    """
    Y_mean = jnp.mean(Y, axis=0)
    interaction = jnp.mean(jnp.einsum('ij,ik->ijk', Y, Y), axis=0)
    
    # Log-partition (approximation via norme L2 pour éviter l'intractabilité)
    log_Z_theta = jnp.sum(jnp.log(1 + jnp.exp(theta @ theta.T)))

    # Calcul de -f(θ)
    f_theta = -log_Z_theta + jnp.sum(theta @ Y_mean.T) + jnp.sum(interaction * theta)
    return f_theta

def wolff_sampler(theta, Y, num_samples=100):
    """
    Wolff sampling algorithm for generating samples from a Boltzmann distribution.
    """
    def single_sample(theta):
        for j in range(theta.shape[0]):
            for k in range(j + 1, theta.shape[1]):
                theta_new = jnp.copy(theta)
                theta_new = theta_new.at[j, k].set(1 - theta[j, k])  # Flip element
                prob_accept = jnp.exp(log_likelihood_penalized(theta_new, Y) - log_likelihood_penalized(theta, Y))
                if np.random.rand() < prob_accept:
                    theta = theta.at[j, k].set(theta_new[j, k])
        return theta


    def single_sample_wrapper(theta):
        return single_sample(theta)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(single_sample_wrapper, theta) for _ in range(num_samples)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    samples = jnp.stack(results)
    return samples

@jit
def prox_g(theta, gamma, lambda_reg, mu_reg):
    """
    Proximal operator for g(θ):
    L1-penalty on off-diagonal entries and L2 on diagonal entries.
    """
    diag = jnp.diag(theta)  # Extract diagonal
    off_diag = theta - jnp.diag(diag)  # Extract off-diagonal entries
    
    # Apply soft-thresholding for L1 penalty on off-diagonal
    prox_off_diag = jnp.sign(off_diag) * jnp.maximum(jnp.abs(off_diag) - gamma * lambda_reg, 0)
    
    # Apply shrinkage for L2 penalty on diagonal
    prox_diag = diag / (1 + gamma * mu_reg)
    
    # Reconstruct matrix
    prox_theta = prox_off_diag + jnp.diag(prox_diag)
    return prox_theta

@jit
def grad_f(theta, Y, num_samples):
    """
    Approximates the gradient of f(θ) using Wolff sampling.
    """

    samples = wolff_sampler(theta, Y, num_samples)
    mean_Y = jnp.mean(Y, axis=0)
    grad = jnp.mean(samples, axis=0) - mean_Y
    return grad


if __name__ == "__main__":
    # Paramètres du modèle
    # Dimensions du problème
    jax.config.update("jax_enable_x64", True)  # Enable double precision
    N = 5  # Nombre d'observations
    p = 2  # Dimension du modèle (p x p)

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


    # Compute gradient
    gradient = grad_f(theta_true, Y, N)
    print("Gradient:", gradient)