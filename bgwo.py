from numba import jit
import numpy as np
import random
import math
from time import time

class BGWO:
    def __init__(self, problem, size=20, max_iterations=100000):
        self.problem = problem
        self.size = size
        self.max_iterations = max_iterations
        
        self.WP = np.zeros((self.size, self.problem.n))
        self.F = np.zeros(self.size)
        self.fa = self.fb = self.fc = 0
        self.a = self.b = self.c = 0
        self.xa = np.zeros(self.problem.n)
        self.xb = np.zeros(self.problem.n)
        self.xc = np.zeros(self.problem.n)
        
        self.init_population()
    
    def init_population(self):
        for k in range(self.size):
            b = -self.problem.b
            for i in self.problem.seq:
                if np.random.rand() < 0.5 and (np.max(b+self.problem.a[:,i]) <= 0):
                    self.WP[k,i] = 1
                    b += self.problem.a[:,i]
                    self.F[k] += self.problem.c[i]
            
            if self.F[k] > self.fa:
                self.a = k
                self.fa = self.F[k]
                self.xa = self.WP[k,:]
            elif self.F[k] > self.fb:
                self.b = k
                self.fb = self.F[k]
                self.xb = self.WP[k,:]
            elif self.F[k] > self.fc:
                self.c = k
                self.fc = self.F[k]
                self.xc = self.WP[k,:]
    
    @jit(forceobj=True)
    def repair(self, x):
        g = np.matmul(self.problem.a, x.reshape(self.problem.n,1)) - self.problem.b.reshape(self.problem.m, 1)
        f = np.matmul(x, self.problem.c)
        for i in reversed(self.problem.seq):
            if np.any(g>0):
                if x[i] == 1:
                    x[i] = 0
                    g -= self.problem.a[:,i].reshape(self.problem.m, 1)
                    f -= self.problem.c[i]
            else:
                break
        for i in self.problem.seq:
            abc = (g + self.problem.a[:, i])
            if x[i] == 0 and np.all(abc <= 0):
                x[i] = 1
                g += self.problem.a[:,i].reshape(self.problem.m, 1)
                f += self.problem.c[i]
        return (x, f)
    
    def phi(self, y):
        return np.abs(np.tanh(y))
    
    @jit(forceobj=True)
    def optimize(self):
        x = np.zeros(self.problem.n)
        con = np.zeros(self.max_iterations)
        i = 0
        while i < self.max_iterations:
            if i%100 == 0:
                print(i, self.fa)
            w = np.array([[self.fa, self.fb, self.fc]])/(self.fa+self.fb+self.fc)
            u = np.exp(-100*i/self.max_iterations)
            cat = np.concatenate((self.xa.reshape(1, self.problem.n), self.xb.reshape(1, self.problem.n), self.xc.reshape(1, self.problem.n)), axis=0)
            xp = (np.matmul(w, cat) + u * np.random.normal(size=(1, self.problem.n))).reshape(self.problem.n)
            for k in range(self.size):
                for j in range(self.problem.n):
                    r = 2* (2*random.random() - 1)
                    y = xp[j] - r * np.abs(xp[j] - self.WP[k, j])
                    r = np.abs(np.tanh(y))
                    if random.random() < r:
                        x[j] = 1
                    else:
                        x[j] = 0
                x, f = self.repair(x)
                if f > self.F[k]:
                    self.F[k] = f
                    self.WP[k,:] = x
                    
                    if self.F[k] > self.fa:
                        self.fa = self.F[k]
                        self.xa = self.WP[k,:]
                        self.a = k
                        if self.fa >= self.problem.best_known:
                            return (self.fa, self.xa, con)
                    elif self.F[k] > self.fb:
                        self.fb = self.F[k]
                        self.xb = self.WP[k,:]
                        self.b = k
                    if self.F[k] > self.fc:
                        self.fc = self.F[k]
                        self.xc = self.WP[k,:]
                        self.c = k
                elif k != self.a and k != self.b and k != self.c:
                    self.F[k] = f
                    self.WP[k,:] = x
                
                con[i] = self.fa
                i += 1
                
        return (self.fa, self.xa, con)