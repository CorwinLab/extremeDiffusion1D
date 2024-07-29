import sys 
sys.path.append("./dataAnalysis")
from theory import KPZ_mean_fit, KPZ_var_fit
from matplotlib import pyplot as plt
import numpy as np

t = np.geomspace(1e-4, 1e5, num=5000)
y = KPZ_var_fit(t)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([min(t), max(t)])
ax.set_xlabel("t")
ax.set_ylabel("Var(h(0, t))")
ax.plot(t, y)
ax.plot(t, (t/2)**(2/3) * 0.813, ls='--')
ax.plot(t, (t * np.pi)**(1/2) /2, ls='--')
fig.savefig("KPZVarFit.pdf", bbox_inches='tight')

def short_time_mean(t):
    # Eq. 20 in Prolhac and Spohn
    return -t / 24 - 1/2 * np.log(2 * np.pi * t)

def long_time_expected_mean(t):
    # Eq. 27 in Prolhac and Spohn (scaled properly)
    return -1.77 * (t/2)**(1/3) - t/24

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t")
ax.set_ylabel("Mean(h(0, t))")
ax.set_xlim([min(t), max(t)])
ax.plot(t, -KPZ_mean_fit(t), 'b', label='Current Numerics')
ax.plot(t, - short_time_mean(t), 'r', label=r'$-\frac{t}{24} - \frac{1}{2}\ln(2 \pi t)$')
ax.plot(t, - long_time_expected_mean(t), 'g', label=r'$-1.77 \left( \frac{t}{2} \right)^{1/3} - \frac{t}{24}$')
ax.legend()
fig.savefig("KPZMeanFit.pdf", bbox_inches='tight')
