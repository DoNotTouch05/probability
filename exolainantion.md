# Explain the code with three figures

This document describes the progromming with three questions and how I solve them

## 1. In Figure1, show $E_{n}, \forall n \in S$, when $\sigma^{2}_{w}∈ (10^{-3}, 10^{-2}, 10^{-1}).$

First setting the sample sizes $S$, for each exponent from 0 to 6, two values are added to $S$: $1\times10^{exponent} 、
5\times10^{exponent}$
```python
S = []
for exponent in range(7):
    S.append(1 * (10 ** exponent))
    S.append(5 * (10 ** exponent))
S = sorted(set(S))
```
Then generating Gaussian samples and compute $a$
```python
def generate_samples(n, mean=0, variance=1):
    return np.random.normal(mean, np.sqrt(variance), n)

def compute_a(sig4ma2w):
    return 1 / (1 + sigma2w)
```
how to get $a$
```
Zt=Xt+Wt, therefore E[Xt*Zt] = E[Xt(Xt+Wt)] = E[Xt^2]+E[Xt]E[Wt]
E[Zt^2] = E[Xt^2]+2E[Xt]E[Wt]+E[Wt^2]
E[Xt] = E[Wt] = 0 and E[Xt^2] = Var(Xt) - E[Xt]^2 = 1 and E[Wt^2] = sigma^2w
divide it and get a = 1 / 1+sigm^2w
```
Next calculating $E_{n}$, and call the $\sigma^{2}_{w}$
```python
def compute_En(a, n, sigma2w):
    Xt = generate_samples(n, 0, 1)
    Wt = generate_samples(n, 0, sigma2w) 
    Zt = Xt + Wt
    X_hat = a * Zt
    En = np.mean((X_hat - Xt) ** 2)
    return En

sigma2w_values = [10**-3, 10**-2, 10**-1]
```
Final plotting the result
```python
plt.figure(figsize=(10, 6))

for sigma2w in sigma2w_values:
    a = compute_a(sigma2w)
    
    En_values = []
    for n in S:
        En = compute_En(a, n, sigma2w)
        En_values.append(En)
    
    plt.semilogx(S, En_values, '-o', label=f'σ²_w = {sigma2w}')
    
plt.xticks(S, labels=[str(val) for val in S])

plt.xlabel('Sample Size (log scale)')
plt.ylabel('Mean Squared Error (En)')
plt.title('Mean Squared Error (En) vs Sample Size (Log Scale)')
plt.legend()
plt.show()
```

## 2. In Figure2, show $E_{n}, \forall n \in S$, when $\sigma^{2}_{w}∈ (10^{-3}, 10^{-2}, 10^{-1}).$
Let $\lambda$ be a positive real number. Consider the case in which $W_{t}$ has the following probability density function.

$$
f_{Wt}(w)
\begin{cases}
\cfrac \lambda2 e^{-\lambda x}, &\forall x\ge0\\
\cfrac \lambda2 e^{\lambda x}, &\forall x\lt0
\end{cases}
$$

Since we have $f_{Wt}(w)$ the PDF function, adding the library to do integration
```python
from scipy.integrate import quad
```
Same as above with the sample sizes $S$
```python
S = []
for exponent in range(7):
    S.append(1 * (10 ** exponent))
    S.append(5 * (10 ** exponent))
S = sorted(set(S))
```
Generating Gaussian and PDF of $W_{t}$
```python
def generate_samples(n, mean=0, variance=1):
    return np.random.normal(mean, np.sqrt(variance), n)

def fWt(w, lambda_val):
    if w >= 0:
        return lambda_val / 2 * np.exp(-lambda_val * w)
    else:
        return lambda_val / 2 * np.exp(lambda_val * w)
```

$W_{t}$ samples and computing $a$ with integration
```python
def generate_Wt_samples(n, lambda_val):
    u = np.random.uniform(0, 1, n)
    w = np.where(u < 0.5, -np.log(1 - 2 * u) / lambda_val, np.log(2 * (u - 0.5)) / lambda_val)
    return w

def compute_a(lambda_val, sigma2w):
    numerator_integral, _ = quad(lambda x: x * (x + np.sqrt(sigma2w)) * np.exp(-x**2 / 2) * fWt(x, lambda_val), -np.inf, np.inf)
    denominator_integral, _ = quad(lambda x: (x + np.sqrt(sigma2w))**2 * fWt(x, lambda_val), -np.inf, np.inf)
    return numerator_integral / denominator_integral
```
Next computing $E_{n}$, call the $\sigma^{2}_{w}$, and set $\lambda$ = 1
```python
def compute_En(a, n, lambda_val, sigma2w):
    Xt = generate_samples(n, 0, 1)
    Wt = generate_Wt_samples(n, lambda_val)
    Zt = Xt + Wt
    X_hat = a * Zt
    En = np.mean((X_hat - Xt) ** 2)
    return En

sigma2w_values = [10**-3, 10**-2, 10**-1]
lambda_val = 1
```
Then plot result
```python
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
```
## 3. In Figure3, show $E'\_{n}$ = $\frac1n \Sigma_{t=1}^n (\hat{X_{t}} - X_{t})^4$ for the two probability density functions of $W_{t}$, when $\sigma^{2}_{w}= 0.16.$
Same step in question 2 adding the library to do integration, setting sample sizes $S$, Gaussian and PDF
```python
from scipy.integrate import quad

S = []
for exponent in range(7):
    S.append(1 * (10 ** exponent))
    S.append(5 * (10 ** exponent))
S = sorted(set(S))

def generate_samples(n, mean=0, variance=1):
    return np.random.normal(mean, np.sqrt(variance), n)

def fWt(w, lambda_val):
    if w >= 0:
        return lambda_val / 2 * np.exp(-lambda_val * w)
    else:
        return lambda_val / 2 * np.exp(lambda_val * w)
```
Also the $W_{t}$ samples and computing $a$ with integration
```python
def generate_specific_samples(n, lambda_val):
    u = np.random.uniform(0, 1, n)
    samples = np.where(u < 0.5, -np.log(2 * u) / lambda_val, np.log(2 * (1 - u)) / lambda_val)
    return samples

def compute_a(lambda_val, sigma2w, fWt_func):
    w = np.sqrt(sigma2w)  # Standard deviation of Wt
    
    def integrand_numerator(x):
        return x * (x + w) * np.exp(-x**2 / 2) * fWt_func(x, lambda_val)

    def integrand_denominator(x):
        return (x + w)**2 * fWt_func(x, lambda_val)
    
    numerator_integral, _ = quad(integrand_numerator, -np.inf, np.inf)
    denominator_integral, _ = quad(integrand_denominator, -np.inf, np.inf)
    
    return numerator_integral / denominator_integral
```
Now we will compute $E'\_{n}$ twice with $W_{t}$ in Gaussian and PDF
```python
def compute_E4n_gaussian(a, n, sigma2w):
    Xt = generate_samples(n, 0, 1)
    Wt = generate_samples(n, 0, sigma2w)
    Zt = Xt + Wt
    X_hat = a * Zt
    E4n = np.mean(np.abs(X_hat - Xt) ** 4)
    return E4n

def compute_E4n_combined(a, n, lambda_val, sigma2w):
    Xt = generate_samples(n, 0, 1)  # Generate samples for Xt
    Wt = generate_specific_samples(n, lambda_val)  # Generate samples for Wt from specific distribution
    Zt = Xt + Wt  # Compute Zt
    X_hat = a * Zt  # Estimated Xt
    E4n = np.mean(np.abs(X_hat - Xt) ** 4)  # Calculate fourth power of absolute estimation error and then take mean
    return E4n
```
The vaiance and $\lambda$
```python
sigma2w = 0.16
lambda_val = 1
```
Different $a$ and $E'\_{n}$ in Gaussian and PDF
```python
a_gaussian = 1 / (1 + sigma2w)
a_combined = compute_a(lambda_val, sigma2w, fWt)

E4n_values_gaussian = [compute_E4n_gaussian(a_gaussian, n, sigma2w) for n in S]
E4n_values_combined = [compute_E4n_combined(a_combined, n, lambda_val, sigma2w) for n in S]
```
Plot
```python
plt.figure(figsize=(10, 6))
plt.semilogx(S, E4n_values_gaussian, '-o', label='Gaussian')
plt.semilogx(S, E4n_values_combined, '-o', label='PDF')
plt.xticks(S, labels=[str(val) for val in S])

plt.xlabel('Sample Size (log scale)')
plt.ylabel('Fourth Power of Absolute Estimation Error (E\'_n)')
plt.title('Fourth Power of Absolute Estimation Error (E\'_n) vs Sample Size (Log Scale)')
plt.legend()
plt.show()
```
