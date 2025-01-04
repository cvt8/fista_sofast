import numpy as np # type: ignore
from grad import grad_f, prox_g

# FISTA algorithm using the paper
# "A Fast Iterative Shrinkage-Thresholding Algorithm"


def fista_use_paper(A, y, lambda_, max_iter=1000, tol=1e-4):
    # A: dictionary
    # y: observation
    # lambda_: regularization parameter
    # max_iter: maximum number of iterations
    # tol: tolerance for stopping criterion
    # return: x_hat, the estimated signal

    # initialization
    x_hat = np.zeros(A.shape[1])
    t = 1
    z = x_hat
    L = np.linalg.norm(A, ord=2)**2
    alpha = 1/L

    # main loop
    for i in range(max_iter):
        x_old = x_hat
        z_old = z

        # update x
        x_hat = z - alpha * A.T @ (A @ z - y)
        x_hat = np.sign(x_hat) * np.maximum(np.abs(x_hat) - alpha * lambda_, 0)

        # update t
        t_old = t
        t = (1 + np.sqrt(1 + 4 * t**2)) / 2

        # update z
        z = x_hat + (t_old - 1) / t * (x_hat - x_old)

        # stopping criterion
        if np.linalg.norm(x_hat - x_old) < tol:
            break

    return x_hat

def p_pg(theta_init, Y, max_iter=2000, gamma=0.1):
    """
    Implements the P-PG algorithm.
    """
    theta = theta_init
    for k in range(max_iter):
        grad = grad_f(theta, Y, num_samples=N)  # Adjust samples
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
        grad = grad_f(y, Y, num_samples=N)
        theta_new = prox_g(y - gamma * grad, gamma)
        theta_old = np.copy(theta)
        theta = np.copy(theta_new)
        t += 1
    print("p_fista", theta)
    return theta