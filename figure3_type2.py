import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Construct the set S
S = []
for exponent in range(7):  # Exponents from 10^0 to 10^6
    S.append(1 * (10 ** exponent))
    S.append(5 * (10 ** exponent))
S = sorted(set(S))  # Remove duplicates and sort

# Function to generate Gaussian samples
def generate_samples(n, mean=0, variance=1):
    return np.random.normal(mean, np.sqrt(variance), n)

# Combined probability density function for Wt
def fWt_combined(w, lambda_val):
    if w >= 0:
        return lambda_val / 2 * np.exp(-lambda_val * w)
    else:
        return lambda_val / 2 * np.exp(lambda_val * w)

# Function to compute 'a' using numerical integration
def compute_a(lambda_val, sigma2w):
    def integrand(x, lambda_val):
        return x * (x + w) * np.exp(-x**2 / 2) * fWt_combined(x, lambda_val)

    numerator_integral, _ = quad(integrand, -np.inf, np.inf, args=(lambda_val,))
    
    def integrand_denominator(x, lambda_val):
        return (x + w)**2 * fWt_combined(x, lambda_val)
    
    denominator_integral, _ = quad(integrand_denominator, -np.inf, np.inf, args=(lambda_val,))
    
    return numerator_integral / denominator_integral

# Function to compute E'^4_n for Gaussian distribution
def compute_E4n_gaussian(a, n, sigma2w):
    Xt = generate_samples(n, 0, 1)  # Generate samples for Xt
    Wt = generate_samples(n, 0, np.sqrt(sigma2w))  # Generate samples for Wt with mean 0 and std dev sqrt(sigma2w)
    Zt = Xt + Wt  # Compute Zt
    X_hat = a * Zt  # Estimated Xt
    E4n = np.mean(np.abs(X_hat - Xt) ** 4)  # Calculate fourth power of absolute estimation error and then take mean
    return E4n

# Function to compute E'^4_n for combined probability density function
def compute_E4n_combined(a, n, lambda_val, sigma2w):
    Xt = generate_samples(n, 0, 1)  # Generate samples for Xt
    Wt = generate_samples(n, 0, np.sqrt(sigma2w))  # Generate samples for Wt with mean 0 and std dev sqrt(sigma2w)
    Zt = Xt + Wt  # Compute Zt
    X_hat = a * Zt  # Estimated Xt
    E4n = np.mean(np.abs(X_hat - Xt) ** 4)  # Calculate fourth power of absolute estimation error and then take mean
    return E4n

# Define noise variance
sigma2w = 0.16
lambda_val = 1  # Value of lambda for the probability density function

# Compute 'a' using numerical integration
w = np.sqrt(sigma2w)  # Standard deviation of Wt
a = compute_a(lambda_val, sigma2w)

# Compute E'^4_n for each sample size in S for Gaussian distribution
E4n_values_gaussian = []
for n in S:
    E4n_gaussian = compute_E4n_gaussian(a, n, sigma2w)
    E4n_values_gaussian.append(E4n_gaussian)

# Compute E'^4_n for each sample size in S for combined probability density function
E4n_values_combined = []
for n in S:
    E4n_combined = compute_E4n_combined(a, n, lambda_val, sigma2w)
    E4n_values_combined.append(E4n_combined)

# Create a figure
plt.figure(figsize=(10, 6))

# Plot E'^4_n for Gaussian distribution
plt.semilogx(S, E4n_values_gaussian, '-o', label='Gaussian')

# Plot E'^4_n for combined probability density function
plt.semilogx(S, E4n_values_combined, '-o', label='Combined PDF')

# Customize x-axis ticks to ensure accuracy
plt.xticks(S, labels=[str(val) for val in S])

# Plot settings
plt.xlabel('Sample Size (log scale)')
plt.ylabel('Fourth Power of Absolute Estimation Error (E\'^4_n)')
plt.title('Fourth Power of Absolute Estimation Error (E\'^4_n) vs Sample Size (Log Scale)')
plt.legend()
plt.show()
