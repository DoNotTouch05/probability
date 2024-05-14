import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Set S (sample sizes)
S = []
for exponent in range(7):
    S.append(1 * (10 ** exponent))
    S.append(5 * (10 ** exponent))
S = sorted(set(S))

# Generate Gaussian samples
def generate_samples(n, mean=0, variance=1):
    return np.random.normal(mean, np.sqrt(variance), n)

# Generate Non-Gaussian samples for Wt based on the given PDF
def generate_Wt_samples(n, lambda_val):
    u = np.random.uniform(0, 1, n)
    Wt = np.zeros(n)
    for i in range(n):
        if u[i] < 0.5:
            Wt[i] = -np.log(1 - 2 * u[i]) / lambda_val
        else:
            Wt[i] = np.log(2 * (u[i] - 0.5)) / lambda_val
    return Wt

# PDF Wt
def fWt(w, lambda_val):
    if w >= 0:
        return lambda_val / 2 * np.exp(-lambda_val * w)
    else:
        return lambda_val / 2 * np.exp(lambda_val * w)

# Compute 'a' using integration
def compute_a(lambda_val, sigma2w):
    w = np.sqrt(sigma2w)
    numerator_integral, _ = quad(lambda x: x * (x + w) * np.exp(-x**2 / 2) * fWt(x, lambda_val), -np.inf, np.inf)
    denominator_integral, _ = quad(lambda x: (x + w)**2 * fWt(x, lambda_val), -np.inf, np.inf)
    return numerator_integral / denominator_integral

# Compute En (Mean Squared Error)
def compute_En(a, n, lambda_val, sigma2w):
    Xt = generate_samples(n, 0, 1)  # Gaussian samples for Xt
    Wt = generate_Wt_samples(n, lambda_val)  # Non-Gaussian samples for Wt
    Zt = Xt + Wt  # Compute Zt
    X_hat = a * Zt  # Estimated Xt
    En = np.mean((X_hat - Xt) ** 2)  # Calculate MSE
    return En

# Variances with sigma2w
sigma2w_values = [10**-3, 10**-2, 10**-1]
lambda_val = 1  # Lambda for PDF

plt.figure(figsize=(10, 6))

for sigma2w in sigma2w_values:
    w = np.sqrt(sigma2w)
    a = compute_a(lambda_val, sigma2w)
    
    En_values = []
    for n in S:
        En = compute_En(a, n, lambda_val, sigma2w)
        En_values.append(En)
    
    plt.semilogx(S, En_values, '-o', label=f'σ²_w = {sigma2w}')
    
plt.xticks(S, labels=[str(val) for val in S])

plt.xlabel('Sample Size (log scale)')
plt.ylabel('Mean Squared Error (En)')
plt.title('Mean Squared Error (En) vs Sample Size (Log Scale)')
plt.legend()
plt.show()
