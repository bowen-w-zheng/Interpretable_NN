import numpy as np
from scipy.optimize import minimize

# Helper functions to vectorize and devectorize matrices
def vec(matrix):
    return matrix.flatten()

def devec(vector, shape):
    return vector.reshape(shape)


def matrix_factorization(X,F, k:int, max_iter:int=1000, tol:float=1e-4, lambda_1:float = 0.5):
    """
    X: result matrix, p by n  
    F: feature matrix, f by n
    k: number of factors
    max_iter: maximum number of iterations
    tol: tolerance for convergence
    lambda: regularization parameter, higher lambda means stronger encouragement for orthogonality
    """
    # Initialize W and H
    p, n = X.shape
    f = F.shape[0]
    # check if n matches second dimension of F, if not print out error message
    assert n == F.shape[1], "second dimension of X and F must match"

    W_shape = (p, k)
    H_shape = (k, f)

    W_init = np.random.rand(*W_shape)
    H_init = np.random.rand(*H_shape)

    # Objective function to minimize
    def objective(params, X, F, W_shape, H_shape):
        W_vec, H_vec = np.split(params, [np.prod(W_shape)])
        W = devec(W_vec, W_shape)
        H = devec(H_vec, H_shape)
        
        # Reconstruction error term
        reconstruction_error = np.linalg.norm(X - W @ H @ F)
        
        # Orthogonality encouraging term
        # Higher lambda means stronger encouragement for orthogonality
        W_orthogonality = np.sum(np.abs(W.T @ W - np.eye(W_shape[1])))
        H_orthogonality = np.sum(np.abs(H @ H.T - np.eye(H_shape[0])))

        return reconstruction_error + lambda_1 * (W_orthogonality + H_orthogonality)
    

    # Run the optimization
    result = minimize(
        objective,
        x0=vec(np.concatenate((W_init, H_init), axis=None)),
        args=(X, F, W_shape, H_shape),
        method='CG', 
        options={'maxiter': max_iter, 'disp': True, 'gtol': tol}
        )
    # Extract the optimized W and H
    W_opt, H_opt = np.split(result.x, [np.prod(W_shape)])
    W_opt = devec(W_opt, W_shape)
    H_opt = devec(H_opt, H_shape)
    return W_opt, H_opt
