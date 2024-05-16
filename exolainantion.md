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

11