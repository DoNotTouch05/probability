import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# set S
S = []
for exponent in range(7):
    S.append(1 * (10 ** exponent))
    S.append(5 * (10 ** exponent))
S = sorted(set(S))

# PDF Wt
def fWt(w, lambda_val):
    if w >= 0:
        return lambda_val / 2 * np.exp(-lambda_val * w)
    else:
        return lambda_val / 2 * np.exp(lambda_val * w)

# compute a integration
def compute_a(lambda_val, sigma2w):
    def numerator_integrand(x, w, lambda_val):
        return x * (x + w) * np.exp(-x**2 / 2) * fWt(w, lambda_val)

    def denominator_integrand(x, w, lambda_val):
        return (x + w)**2 * fWt(w, lambda_val)

    w = np.sqrt(sigma2w)
    numerator_integral, _ = quad(lambda x: numerator_integrand(x, w, lambda_val), -np.inf, np.inf)
    denominator_integral, _ = quad(lambda x: denominator_integrand(x, w, lambda_val), -np.inf, np.inf)
    return numerator_integral / denominator_integral

# compute En
def compute_En(a, n, lambda_val, sigma2w):
    Xt = np.random.normal(0, 1, n)
    Wt = np.array([np.random.exponential(scale=1/lambda_val) if np.random.rand() < 0.5 else -np.random.exponential(scale=1/lambda_val) for _ in range(n)])
    Zt = Xt + Wt  # Compute Zt
    X_hat = a * Zt  # Estimated Xt
    En = np.mean((X_hat - Xt) ** 2)  # Calculate MSE
    return En

# variances with sigma2w
sigma2w_values = [10**-3, 10**-2, 10**-1]
lambda_val = 1  # lambda for PDF

plt.figure(figsize=(10, 6))

for sigma2w in sigma2w_values:
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
