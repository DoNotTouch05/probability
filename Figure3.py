import numpy as np
import matplotlib.pyplot as plt

# Define the set S
S = []
for exponent in range(7):  # Exponents from 10^0 to 10^6
    S.append(1 * (10 ** exponent))
    S.append(5 * (10 ** exponent))
S = sorted(set(S))  # Remove duplicates and sort

# Generate Gaussian samples
def generate_samples(n, mean=0, variance=1):
    return np.random.normal(mean, np.sqrt(variance), n)

# Generate samples from the combined exponential distribution
def generate_samples_exponential(n, lambda_val):
    samples = np.random.exponential(1 / lambda_val, n)  # Generate exponential samples
    signs = np.random.choice([-1, 1], n)  # Randomly assign positive or negative signs
    return samples * signs  # Apply signs to the samples

# Compute the value of 'a' for Gaussian distribution
def compute_a_gaussian(sigma2w):
    return 1 / (1 + sigma2w)  # Derived formula for 'a'

# Compute the value of 'a' for combined exponential distribution
def compute_a_exponential(lambda_val, sigma2w):
    return 1 / (1 + sigma2w)  # Same derived formula applies because of symmetry and zero mean

# PDF for combined exponential distribution
def fWt_combined(w, lambda_val):
    if w >= 0:
        return (lambda_val / 2) * np.exp(-lambda_val * w)
    else:
        return (lambda_val / 2) * np.exp(lambda_val * w)

# Compute E_n for Gaussian
def compute_En_gaussian(a, n, sigma2w):
    Xt = generate_samples(n, 0, 1)
    Wt = generate_samples(n, 0, sigma2w)
    Zt = Xt + Wt
    X_hat = a * Zt
    En = np.mean((X_hat - Xt) ** 4)
    return En

# Compute E_n for combined exponential distribution
def compute_En_exponential(a, n, lambda_val):
    Xt = generate_samples(n, 0, 1)
    Wt = generate_samples_exponential(n, lambda_val)
    Zt = Xt + Wt
    X_hat = a * Zt
    En = np.mean((X_hat - Xt) ** 4)
    return En

# Variance and lambda values
sigma2w = 0.16
lambda_val = 1  # Lambda for PDF

# Compute 'a' values for both distributions
a_gaussian = compute_a_gaussian(sigma2w)
a_exponential = compute_a_exponential(lambda_val, sigma2w)

# Compute E_n for each sample size in S for Gaussian
En_values_gaussian = []
for n in S:
    En_gaussian = compute_En_gaussian(a_gaussian, n, sigma2w)
    En_values_gaussian.append(En_gaussian)

# Compute E_n for each sample size in S for combined exponential distribution
En_values_exponential = []
for n in S:
    En_exponential = compute_En_exponential(a_exponential, n, lambda_val)
    En_values_exponential.append(En_exponential)

# Plotting the results
plt.figure(figsize=(10, 6))

plt.semilogx(S, En_values_gaussian, '-o', label='Gaussian')
plt.semilogx(S, En_values_exponential, '-o', label='Exponential')

plt.xticks(S, labels=[str(val) for val in S])

plt.xlabel('Sample Size (log scale)')
plt.ylabel("Mean Squared Error (E_n)")
plt.title("Mean Squared Error (E_n) vs Sample Size (Log Scale)")
plt.legend()
plt.show()
