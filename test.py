import time 
import sys 
sys.path.append("./src")
from pydiffusionPDF import DiffusionPDF 
import numpy as np
import npquad 

if __name__ == "__main__": 
    d = DiffusionPDF(np.quad("1e300"), 1, int(1e7))
    s = time.process_time()
    d.evolveAndSaveQuantiles(list(range(1, 1000)), [10, 100], "Quantiles.txt") 
    print(time.process_time() - s)

