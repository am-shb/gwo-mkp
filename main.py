from problem import Problem
from bgwo import BGWO

p = Problem('GK MKP Benchmarks/gk11.dat')

bgwo = BGWO(p)
opt = bgwo.optimize()