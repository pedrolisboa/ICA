import numpy as np
from scipy import signal


def toy_time(n_samples, noise_amp=0.2, num_spike_outliers=0):
    time = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3 : saw tooth signal
    
    S = np.c_[s1, s2, s3]
    S += noise_amp * np.random.normal(size=S.shape)  # Add noise
    
    S /= S.std(axis=0)  # Standardize data
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    X = np.dot(S, A.T)  # Mixed signals

    if num_spike_outliers > 0:
        outlier_indices = np.random.choice(n_samples, num_spike_outliers, replace=False)
        outlier_magnitudes = np.random.uniform(low=-10, high=10, size=(num_spike_outliers, 3))
        for idx, mag in zip(outlier_indices, outlier_magnitudes):
            X[idx] += mag
    
    return S, A, X