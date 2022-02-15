import numpy as np
import pandas as pd
import npquad
from matplotlib import pyplot as plt

df = pd.read_csv("/home/jacob/Desktop/corwinLabMount/CleanData/Paper/CDF/Quartiles0.txt")
df = df.drop_duplicates(subset=['time'], keep='last')
df.reset_index(inplace=True)

fig, ax = plt.subplots()
ax.plot(df['time'])
fig.savefig("time.png")
