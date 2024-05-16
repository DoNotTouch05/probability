import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Construct the set S
S = []
for exponent in range(7): 
    S.append(1 * (10 ** exponent))
    S.append(5 * (10 ** exponent))
S = sorted(set(S)) 

# Gaussian samples
def generate_samples(n, mean=0, variance=1):
    return np.random.normal(mean, np.sqrt(variance), n)

# PDF Wt
def fWt(w, lambda_val):
    if w >= 0:
        return lambda_val / 2 * np.exp(-lambda_val * w)
    else:
        return lambda_val / 2 * np.exp(lambda_val * w)

# generate samples PDF
def generate_specific_samples(n, lambda_val):
    u = np.random.uniform(0, 1, n)
    samples = np.where(u < 0.5, -np.log(2 * u) / lambda_val, np.log(2 * (1 - u)) / lambda_val)
    return samples

# compute a integration
def compute_a(lambda_val, sigma2w, fWt_func):
    w = np.sqrt(sigma2w)  # Standard deviation of Wt
    
    def integrand_numerator(x):
        return x * (x + w) * np.exp(-x**2 / 2) * fWt_func(x, lambda_val)

    def integrand_denominator(x):
        return (x + w)**2 * fWt_func(x, lambda_val)
    
    numerator_integral, _ = quad(integrand_numerator, -np.inf, np.inf)
    denominator_integral, _ = quad(integrand_denominator, -np.inf, np.inf)
    
    return numerator_integral / denominator_integral

# compute E'^4_n in Gaussian distribution
def compute_E4n_gaussian(a, n, sigma2w):
    Xt = generate_samples(n, 0, 1)  # Generate samples for Xt
    Wt = generate_samples(n, 0, sigma2w)  # Generate samples for Wt with mean 0 and variance sigma2w
    Zt = Xt + Wt  # Compute Zt
    X_hat = a * Zt  # Estimated Xt
    E4n = np.mean(np.abs(X_hat - Xt) ** 4)  # Calculate fourth power of absolute estimation error and then take mean
    return E4n

# compute E'^4_n for Wt with PDF
def compute_E4n_combined(a, n, lambda_val, sigma2w):
    Xt = generate_samples(n, 0, 1) 
    Wt = generate_specific_samples(n, lambda_val)  
    Zt = Xt + Wt 
    X_hat = a * Zt 
    E4n = np.mean(np.abs(X_hat - Xt) ** 4) 
    return E4n

# variance
sigma2w = 0.16
lambda_val = 1 

# Compute a for Gaussian
a_gaussian = 1 / (1 + sigma2w)

# Compute a for PDF Wt
a_combined = compute_a(lambda_val, sigma2w, fWt)

# Compute E'^4_n for each sample size in Gaussian
E4n_values_gaussian = [compute_E4n_gaussian(a_gaussian, n, sigma2w) for n in S]

# Compute E'^4_n for each sample size in PDF
E4n_values_combined = [compute_E4n_combined(a_combined, n, lambda_val, sigma2w) for n in S]

# Plot settings
plt.figure(figsize=(10, 6))

plt.semilogx(S, E4n_values_gaussian, '-o', label='Gaussian')

plt.semilogx(S, E4n_values_combined, '-o', label='PDF')

plt.xticks(S, labels=[str(val) for val in S])


plt.xlabel('Sample Size (log scale)')
plt.ylabel('Fourth Power of Absolute Estimation Error (E\'_n)')
plt.title('Fourth Power of Absolute Estimation Error (E\'_n) vs Sample Size (Log Scale)')
plt.legend()
plt.show()
