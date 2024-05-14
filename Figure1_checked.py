import numpy as np
import matplotlib.pyplot as plt

'''
Z=X+W, X*(X+W)=X^2+XW and (X+W)^2=X^2+2XW+W^2
E[X]=E[W]=0, E[XZ]=E[X^2]=1, E[Z^2]=E[X^2]+E[W^2]=1+sigma2w
'''

#set S
S = []
for exponent in range(7):  # Exponents from 10^0 to 10^6
    S.append(1 * (10 ** exponent))
    S.append(5 * (10 ** exponent))
S = sorted(set(S))  # Remove duplicates and sort

#Gaussian samples
def generate_samples(n, mean=0, variance=1):
    return np.random.normal(mean, np.sqrt(variance), n)

#compute a
def compute_a(sigma2w):
    return 1 / (1 + sigma2w)  # Derived formula for 'a'

#compute En
def compute_En(a, n, sigma2w):
    Xt = generate_samples(n, 0, 1)
    Wt = generate_samples(n, 0, sigma2w) 
    Zt = Xt + Wt  #Zt
    X_hat = a * Zt
    En = np.mean((X_hat - Xt) ** 2)
    return En

#variances with sigma2w
sigma2w_values = [10**-3, 10**-2, 10**-1]

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
