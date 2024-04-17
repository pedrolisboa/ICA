import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma,psi
from sklearn.neighbors import KernelDensity

EPS = np.finfo(float).eps

"""
Author: Micael Veríssimo de Araújo
"""


def amari_error(A, B, squares=False):
    """
    Calculates the performance E1 or E2 as in Oja's book. E2 is square = true, otherwise E1 is compute
    A is the mixture matrix and B is the estimate mixture matrix
    """

    P = np.dot(B,A)

    if squares:
        P = P**2
    else:
        P = np.abs(P)

    max_lines = np.max(P, axis=1)
    max_cols  = np.max(P, axis=0)
    sum_lines = np.sum(np.sum(P/np.tile(max_lines, [P.shape[0], 1]).T, axis=1) -1)
    sum_cols  = np.sum(np.sum(P/np.tile(max_cols , [P.shape[1], 1])  , axis=0) -1)

    return sum_lines + sum_cols


# Estimating PDF
def EstPDF(data, bins=np.array([-1,0, 1]), mode='kernel', kernel='epanechnikov', kernel_bw=0.01):
    # kernels = 'epanechnikov','gaussian', 'tophat','exponential', 'linear', 'cosine'
    if mode == 'hist':
        #print 'EstPDF: Histogram Mode'
        [y,pts] = np.histogram(data,bins=bins,density=True)
        bins_centers = pts[0:-1]+np.diff(pts)
        pdf = y*np.diff(pts)
        return [pdf,bins_centers]
    if mode == 'kernel':
        print('EstPDF: Kernel Mode')
        if kernel is None:
            print('No kernel defined')
            return -1
        if kernel_bw is None:
            print('No kernel bandwidth defined')
            return -1
        kde = (KernelDensity(kernel=kernel,algorithm='auto',bandwidth=kernel_bw).fit(data))
        aux_bins = bins
        log_dens_x = (kde.score_samples(aux_bins[:, np.newaxis]))
        pdf = np.exp(log_dens_x)
        pdf = pdf/sum(pdf)
        bins_centers = bins
        return [pdf,bins_centers]

def nearest_distances(X, k=1):
    '''
        X = array(N,M)
        N = number of points
        M = number of dimensions
        returns the distance to the kth nearest neighbor for every point in X
    '''
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    d, _ = knn.kneighbors(X) # the first nearest neighbor is itself
    return d[:, -1] # returns the distance to the kth nearest neighbor


def entropy(X, k=1):
    ''' 
        Returns the entropy of the X.
        Parameters
        ===========
        X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed
        k : int, optional
        number of nearest neighbors for density estimation
        Notes
        ======
        Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
        of a random vector. Probl. Inf. Transm. 23, 95-101.
        See also: Evans, D. 2008 A computationally efficient estimator for
        mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
        and:
        Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
        information. Phys Rev E 69(6 Pt 2):066138.
    '''
    
    # Distance to kth nearest neighbor
    X = X[:,np.newaxis]
    r = nearest_distances(X, k) # squared distances
    n, d = X.shape
    volume_unit_ball = (np.pi**(.5*d)) / gamma(.5*d + 1)
    '''
        F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
        for Continuous Random Variables. Advances in Neural Information
        Processing Systems 21 (NIPS). Vancouver (Canada), December.
        return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
        '''
    return (d*np.mean(np.log(r + np.finfo(X.dtype).eps))
            + np.log(volume_unit_ball) + psi(n) - psi(k))


def kl_divergence(p, q, bins=np.array([-1,0, 1]), mode='kernel', kernel='epanechnikov', kernel_bw=0.1):
    [p_pdf,p_bins] = EstPDF(p, bins=bins, mode='hist')
    [q_pdf,q_bins] = EstPDF(q, bins=bins, mode='hist')
    kl_values = []
    for i in range(len(p_pdf)):
        if p_pdf[i] == 0 or q_pdf[i] == 0 :
            kl_values = np.append(kl_values,0)
        else:
            kl_value = np.abs(p_pdf[i]*np.log10(p_pdf[i]/q_pdf[i]))
            if np.isnan(kl_value):
                kl_values = np.append(kl_values,0)
            else:
                kl_values = np.append(kl_values,kl_value)
    return [np.sum(kl_values),kl_values]



def mutual_information_score(variables, k=1):
    '''
        Returns the mutual information between any number of variables.
        Each variable is a matrix X = array(n_samples, n_features)
        where
        n = number of samples
        dx,dy = number of dimensions
        Optionally, the following keyword argument can be specified:
        k = number of nearest neighbors for density estimation
        Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)
        '''
    if len(variables) < 2:
        raise AttributeError("Mutual information must involve at least 2 variables")
    all_vars = np.hstack(variables)
    return (sum([entropy(X, k=k) for X in variables]) - entropy(all_vars, k=k))