import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erf
import sys

sys.path.append("./DiffusionCDF")
from diffusionCDF import (
    calculateMean,
    calculateVariance,
    calculateMeanPDF,
    calculateVariancePDF,
)
import npquad


def guassian_cdf(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / sigma / np.sqrt(2)))


def gumbel_cdf(x, mu, beta):
    return np.exp(-np.exp(-(x - mu) / beta))


# Make a simple CDF for a guassian. I spaced all the x-values by 1 b/c that's how
# my data is spaced and it's easier to deal with.
mu = 500
sigma = 50
x_vals = np.arange(-int(1e4), int(1e4), 1)
cdf = gumbel_cdf(x_vals, mu, sigma)


def get_variance(x_vals, cdf):
    # ensure all x-vals are > 0
    x_vals = x_vals - x_vals[0]
    assert all(x_vals >= 0)

    first_sum = 0
    second_sum = 0
    for i in range(len(cdf)):
        first_sum += 2 * x_vals[i] * (1 - cdf[i])
        second_sum += 1 - cdf[i]
    var = first_sum - second_sum ** 2

    return var


def get_mean(x_vals, cdf):
    mean_sum = 0
    for i in range(len(cdf)):
        if x_vals[i] <= 0:
            H = 0
        else:
            H = 1
        mean_sum += H - cdf[i]
    return mean_sum


def get_variance_Heaviside(x_vals, cdf):
    mean = get_mean(x_vals, cdf)
    print("Python Calculated Mean:", mean)
    x_vals = x_vals - mean
    first_sum = 0
    second_sum = 0
    for i in range(len(cdf)):
        x = x_vals[i]
        F = cdf[i]
        if x <= 0:
            H = 0
        else:
            H = 1
        first_sum += 2 * x * (H - F)
        second_sum += H - F
    var = first_sum - second_sum ** 2
    print("Mean after shift:", second_sum)
    return var


pdf = np.diff(cdf)
print("Sum PDF:", sum(pdf))
var = get_variance(x_vals, cdf)
print("--------Mean----------------")
print("Actual Mean:", mu)
heavy_var = get_variance_Heaviside(x_vals, cdf)
print("Python mean:", get_mean(x_vals, cdf))
print("C++ mean from CDF:", calculateMean(x_vals, cdf))
print("C++ mean from PDF:", calculateMeanPDF(x_vals[1:], pdf))
print("--------Variance------------")
print("Wikipedia Variance:", var)
print("Heaviside Variance:", heavy_var)
print("C++ variance from CDF:", calculateVariance(x_vals, cdf))
print("C++ variance from PDF:", calculateVariancePDF(x_vals[1:], pdf))
print("Actual Variance: ", np.pi ** 2 / 6 * sigma ** 2)
fig, ax = plt.subplots()
ax.plot(x_vals, cdf, label="CDF")
ax.plot(x_vals[1:], pdf, label="PDF")
ax.grid(True)
ax.legend()
fig.savefig("CDF.png")
