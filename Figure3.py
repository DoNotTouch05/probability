import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Set S
S = []
for exponent in range(7):  # Exponents from 10^0 to 10^6
    S.append(1 * (10 ** exponent))
    S.append(5 * (10 ** exponent))
S = sorted(set(S))

# Gaussian samples
def generate_samples(n, mean=0, variance=1):
    return np.random.normal(mean, np.sqrt(variance), n)

#PDF for Wt
def fWt_combined(w, lambda_val):
    if w >= 0:
        return lambda_val / 2 * np.exp(-lambda_val * w)
    else:
        return lambda_val / 2 * np.exp(lambda_val * w)
      
#compute 'a'
def compute_a(lambda_val, sigma2w, distribution='gaussian'):
    # Define the integrands for numerator and denominator
    def integrand(x, lambda_val, distribution):
        if distribution == 'gaussian':
            return x * (x + w) * np.exp(-x**2 / 2)
        elif distribution == 'combined':
            return x * (x + w) * np.exp(-x**2 / 2) * fWt_combined(x, lambda_val)
        else:
            raise ValueError("Invalid distribution type. Use 'gaussian' or 'combined'.")

    def integrand_denominator(x, lambda_val, distribution):
        if distribution == 'gaussian':
            return (x + w)**2
        elif distribution == 'combined':
            return (x + w)**2 * fWt_combined(x, lambda_val)
        else:
            raise ValueError("Invalid distribution type. Use 'gaussian' or 'combined'.")

    numerator_integral, _ = quad(integrand, -np.inf, np.inf, args=(lambda_val, distribution))
    denominator_integral, _ = quad(integrand_denominator, -np.inf, np.inf, args=(lambda_val, distribution))

    return numerator_integral / denominator_integral

#compute E'n for Gaussian 
def compute_E4n_gaussian(a, n, sigma2w):
    Xt = generate_samples(n, 0, 1)  
    Wt = generate_samples(n, 0, np.sqrt(sigma2w))  
    Zt = Xt + Wt  
    X_hat = a * Zt 
    E4n = np.mean(np.abs(X_hat - Xt) ** 4)  

    return E4n

#compute E'n for PDF
def compute_E4n_combined(a, n, lambda_val, sigma2w):
    Xt = generate_samples(n, 0, 1)  
    Wt = generate_samples(n, 0, np.sqrt(sigma2w))  
    Zt = Xt + Wt  
    X_hat = a * Zt  
    E4n = np.mean(np.abs(X_hat - Xt) ** 4)  
    return E4n

#variance
sigma2w = 0.16
lambda_val = 1  #lambda for PDF

#compute 'a'
w = np.sqrt(sigma2w)  # Standard deviation of Wt
a_gaussian = compute_a(lambda_val, sigma2w, distribution='gaussian')
a_combined = compute_a(lambda_val, sigma2w, distribution='combined')

#compute E'_n for each sample for Gaussian
E4n_values_gaussian = []
for n in S:
    E4n_gaussian = compute_E4n_gaussian(a_gaussian, n, sigma2w)
    E4n_values_gaussian.append(E4n_gaussian)

#compute E'_n for each sample for PDF
E4n_values_combined = []
for n in S:
    E4n_combined = compute_E4n_combined(a_combined, n, lambda_val, sigma2w)
    E4n_values_combined.append(E4n_combined)

plt.figure(figsize=(10, 6))

plt.semilogx(S, E4n_values_gaussian, '-o', label='Gaussian')

plt.semilogx(S, E4n_values_combined, '-o', label='fWt')

plt.xticks(S, labels=[str(val) for val in S])

plt.xlabel('Sample Size (log scale)')
plt.ylabel('Fourth Power of Absolute Estimation Error (E_n)')
plt.title('Fourth Power of Absolute Estimation Error (E_n) vs Sample Size (Log Scale)')
plt.legend()
plt.show()
