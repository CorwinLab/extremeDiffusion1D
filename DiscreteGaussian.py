import numpy as np
from matplotlib import pyplot as plt


def quantile(arr, quantile):
    num_in_arr = sum(arr)
    run_sum = 0
    for i in range(len(arr)):
        if run_sum >= num_in_arr * quantile:
            return i
        run_sum += arr[i]


num_to_generate = int(1e4)
mu, sigma = 2500, 500
N = 1000
prob = []
xs = np.arange(int(mu - sigma * 3), int(mu + sigma * 3))
for _ in range(num_to_generate):
    # Make a discrete guassian distribution
    s = np.around(np.random.normal(mu, sigma, N))

    Mn = max(s)
    prob.append(Mn <= xs)
prob = np.array(prob).astype(int)
mean = np.mean(prob, 0)  # Okay so I think this is F_N(x)

new_probs = []
for _ in range(num_to_generate):
    single_val = np.around(np.random.normal(mu, sigma, 1))[0]
    new_probs.append(single_val <= xs)
new_probs = np.array(new_probs).astype(int)
mean_new = np.mean(new_probs, 0)

PDF = np.diff(mean_new)
PDF_x = xs[1:]
PDF_mean = sum(PDF * PDF_x)
PDF_var = sum(PDF * (PDF_x - PDF_mean) ** 2)
print("Calculated Mean:", PDF_mean)
print("Calculated Variance:", PDF_var)

a = 10
idx10 = quantile(PDF, 1 / a)
idx100 = quantile(PDF, 1 / 10 / a)
print("10N - 0.1N Quantile Variance:", (idx10 - idx100) ** 2)

fig, ax = plt.subplots()
ax.scatter(xs, mean, label=r"$Prob(M_{N} \leq x)$")
ax.scatter(xs, mean_new ** N, label=r"$(Prob(X\leq x))^{N}$")
ax.scatter(PDF_x, PDF, label=r"$f_{N}(x)$")
ax.set_xlabel("x")
ax.set_title("Guassian Distribution with N=10 Particles")
ax.legend()
fig.savefig("CDF.png")
