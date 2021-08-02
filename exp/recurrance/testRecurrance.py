import sys
sys.path.append("../../recuranceRelation")
import recurrance
import numpy as np
import npquad

zB = recurrance.makeRec(10)
zB = np.array(zB, dtype=np.quad)
