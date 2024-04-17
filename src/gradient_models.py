import numpy as np
import numpy as np
from sklearn.decomposition import FastICA, PCA

class GradientICA:
    def __init__(self, n_components, max_iter=1000, learning_rate=0.01, method='negentropy', g_func='g1', approach='deflationary', whiten=True):
        self.n_components = n_components
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.method = method
        self.g_func = g_func
        self.approach = approach
        self.whiten = whiten
        self.weights = None
        self.whitening_matrix = None

    @staticmethod
    def kurtosis(x):
        n = x.shape[0]
        mean_x = np.mean(x)
        fourth_moment = np.mean((x - mean_x)**4)
        variance = np.var(x)
        return (fourth_moment / (variance**2)) - 3

    @staticmethod
    def G1(y, a=1):
        return (1/a) * np.log(np.cosh(a * y))

    @staticmethod
    def G2(y):
        return -np.exp(-0.5 * y**2)

    @staticmethod
    def expected_G(Y, func):
        return np.mean(func(Y))

    @staticmethod
    def negentropy(Y, gaussian_Y, func):
        k = 1
        return k * (GradientICA.expected_G(Y, func) - GradientICA.expected_G(gaussian_Y, func))**2

    def compute_gradient(self, X, weights):
        Z = X @ weights
        if self.method == 'kurtosis':
            raise NotImplementedError
            # mean_Z = np.mean(Z)
            # fourth_moment_grad = 4 * np.mean((Z - mean_Z)**3) * (Z - mean_Z)
            # variance = np.var(Z)
            # second_moment_grad = -6 * variance * np.mean((Z - mean_Z)**2)
            # gradient = X.T @ (fourth_moment_grad + second_moment_grad) / (variance**2)
        else:
            if self.g_func == 'g1':
                psi_Z = np.tanh(Z)  # Derivative of G1
            elif self.g_func == 'g2':
                psi_Z = Z * np.exp(-0.5 * Z**2)  # Derivative of G2
            gradient = Z @ psi_Z
        return gradient / len(Z)

    def whiten_data(self, X):
        # Compute the covariance matrix
        X_mean = np.mean(X, axis=0)
        X1 = X - X_mean
        # covariance_matrix = np.cov(X_centered, rowvar=False)
        # Eigenvalue decomposition of the covariance matrix
        d, u = np.linalg.eigh(X1.T.dot(X1))
        sort_indices = np.argsort(d)[::-1]
        eps = np.finfo(d.dtype).eps
        degenerate_idx = d < eps
        if np.any(degenerate_idx):
            warnings.warn(
                "There are some small singular values"
            )
        d[degenerate_idx] = eps  # For numerical issues
        np.sqrt(d, out=d)
        d, u = d[sort_indices], u[:, sort_indices]

        # Give consistent eigenvectors for both svd solvers
        u *= np.sign(u[0])

        K = (u / d).T#[:n_components]  # see (6.33) p.140
        del u, d
        X1 = np.dot(K, X1.T)
        # see (13.6) p.267 Here X1 is white and data
        # in X has been projected onto a subspace by PCA
        X1 *= np.sqrt(X1.shape[0])
        
        return X1.T
    
    def fit(self, X):
        if self.whiten:
            X = self.whiten_data(X)
        else:
            X -= X.mean()
            
        self.weights = np.random.randn(self.n_components, X.shape[1])
        gamma = np.random.uniform()

        gaussian_Y = np.random.normal(0, 1, X.shape[0]) if self.method == 'negentropy' else None
        
        for _ in range(self.max_iter):
            for i in range(self.n_components):
                gradient = self.compute_gradient(X, self.weights[i, :])
                self.weights[i, :] += self.learning_rate * gradient# * gamma
                # if self.approach == 'deflationary':
                #     for j in range(i):
                #         self.weights[i, :] -= np.dot(self.weights[i, :], self.weights[j, :]) * self.weights[j, :]
                
                self.weights[i, :] /= np.linalg.norm(self.weights[i, :])  # Normalize

                # Update gamma
                if self.method == 'negentropy':
                    Y = X @ self.weights[i, :]
                    gamma += np.mean(self.G1(Y) - GradientICA.expected_G(gaussian_Y, self.G1))

                
            if _ % 100 == 0:
                if self.method == 'kurtosis':
                    metric = self.kurtosis(X @ self.weights[:, i])
                else:
                    Y = X @ self.weights[:, i]
                    metric = self.negentropy(Y, gaussian_Y, self.G1 if self.g_func == 'g1' else self.G2)
                print(f'Iteration {_+1}, Component {i+1}, Metric: {metric}, Gamma: {gamma}')

    def transform(self, X):
        return X @ self.weights.T


class NaturalGradientICA:
    def __init__(self, n_components, max_iter=1000, whiten=True, mu=0.01, mu_gamma=0.01, tol=1e-08):
        self.n_components = n_components
        self.max_iter = max_iter
        self.mu = mu
        self.mu_gamma = mu_gamma
        self.weights = None
        self.tol = tol
        self.whiten = whiten

    def _logcosh(self, x):
        """ Nonlinear function used as a contrast function, g(y). """
        return np.tanh(x)*x, (1 - np.tanh(x)**2)  # g(y) and its derivative g'(y)

    def g_plus(self, x):
        return np.tanh(x)*x

    def whiten_data(self, X):
        # Compute the covariance matrix
        X_mean = np.mean(X, axis=0)
        X1 = X - X_mean
        # covariance_matrix = np.cov(X_centered, rowvar=False)
        # Eigenvalue decomposition of the covariance matrix
        d, u = np.linalg.eigh(X1.T.dot(X1))
        sort_indices = np.argsort(d)[::-1]
        eps = np.finfo(d.dtype).eps
        degenerate_idx = d < eps
        if np.any(degenerate_idx):
            warnings.warn(
                "There are some small singular values"
            )
        d[degenerate_idx] = eps  # For numerical issues
        np.sqrt(d, out=d)
        d, u = d[sort_indices], u[:, sort_indices]

        # Give consistent eigenvectors for both svd solvers
        u *= np.sign(u[0])

        K = (u / d).T#[:n_components]  # see (6.33) p.140
        del u, d
        X1 = np.dot(K, X1.T)
        # see (13.6) p.267 Here X1 is white and data
        # in X has been projected onto a subspace by PCA
        X1 *= np.sqrt(X1.shape[0])
        
        # Compute the whitening matrix
        # self.whitening_matrix = eigen_vectors @ np.diag(1.0 / np.sqrt(eigen_values)) @ eigen_vectors.T
        return X1.T
    
    def fit(self, X):
        num_samples, num_features = X.shape
        if self.whiten:
            X = self.whiten_data(X)
        else:
            X = X - X.mean()
        
        # Initialize weights randomly
        self.weights = np.random.randn(self.n_components, num_features)
        
        # Normalize each row to be of unit norm
        self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True)
        
        self.old_weights = np.copy(self.weights[:, :])
        gamma = np.random.random(size=self.n_components)

        G_fun = np.eye(self.n_components)

        iter = 0
        done = False
        while not done:
            # Estimate the sources
            S = X.dot(self.weights.T)
            
            for i in range(self.n_components):
                y = S[:, i]
                gamma[i] = (1 - self.mu_gamma)*gamma[i] + self.mu_gamma*np.mean(-np.tanh(y)*y + (1-np.tanh(y)**2))
            
            g_fun = list()
            for i in range(self.n_components):
                if gamma[i] > 0:
                    G_fun[i,i] = -2*np.tanh(y[i])
                else:
                    G_fun[i,i] = np.tanh(y[i]) - y[i]
                    
            # Re-normalize weights
            # self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True)
            I = np.eye(self.n_components)
            self.weights -= self.mu*(I + G_fun) @ self.weights
            
            
            wdiff = np.sum(self.weights - self.old_weights)
            # self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True)
            
            if iter % 100 == 0:
                print(f"Iteration {iter}: Weights updated. Diff {wdiff}")
            done = np.abs(wdiff) < self.tol
            done = done or (iter > self.max_iter)
            iter += 1
            self.old_weights = np.copy(self.weights[:, :])



    def transform(self, X):
        """Project X onto the independent components."""
        return X.dot(self.weights.T)
