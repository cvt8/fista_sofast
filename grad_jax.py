import numpy as np

def penalty_g(theta, lambda_reg, mu_reg):
    """
    Calcule la pénalité g(θ) = λ * ||θ_offdiag||_1 + μ * ||diag(θ)||_2^2.
    """
    diag_part = np.diag(theta)**2  # Partie quadratique (diagonale)
    off_diag_part = np.sum(np.abs(theta - np.diag(np.diag(theta))))  # Partie L1 hors diagonale
    res = lambda_reg * off_diag_part + mu_reg * np.sum(diag_part)
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

def wolff_sampler(theta, Y, num_samples=100):
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
                    theta[j, k] = theta_new[j, k]
        samples[i] = theta
    return samples


def prox_g(theta, gamma, lambda_reg, mu_reg):
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
    return prox_theta

def grad_f(theta, Y, num_samples):
    """
    Approximates the gradient of f(θ) using Wolff sampling.
    """
    # Use Wolff sampling to generate samples
    samples = wolff_sampler(theta, Y, num_samples)
    grad = np.zeros_like(theta)
    for sample in samples:
        grad += sample - np.mean(Y, axis=0)
    grad /= num_samples
    return grad



if __name__ == "__main__":
    # Paramètres de simulation
    N = 250  # Nombre d'observations
    p = 100  # Dimension du modèle (p x p)
    np.random.seed(42) # Pour la reproductibilité

    # Génération des données (modèle graphique binaire)
    Y = np.random.binomial(1, 0.5, size=(N, p))  # N échantillons de dimension p

    # Définition de la pénalité g(θ)
    lambda_reg = 0.5 * np.sqrt(np.log(p) / N)  # Paramètre de régularisation L1
    mu_reg = 0.5  # Paramètre de régularisation L2

    # Initialisation aléatoire de θ
    upper_indices = np.triu_indices(p, k=1)  # Indices des éléments hors diagonale (supérieurs)
    num_upper = len(upper_indices[0])  # Nombre total d'éléments hors diagonale
    theta_true = np.zeros((p, p))
    upper_indices = np.triu_indices(p, k=1)  # Indices des éléments hors diagonale (supérieurs)
    theta_true[upper_indices] = np.random.choice([0, 1], size=num_upper, p=[0.98, 0.02])
    np.fill_diagonal(theta_true, np.random.uniform(0.5, 1.0, size=p))  # Diagonale non nulle


    gradient = grad_f(theta_true, Y, N)
    print(gradient)