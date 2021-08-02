import sys
sys.path.append("../../recuranceRelation")
from recurrance import Recurrance
import numpy as np
import npquad

rec = Recurrance(1, 1_000)
rec.makeRec()
q = rec.findQuintile(10)
