import numpy as np
from scipy.optimize import linprog

class Problem:
    '''
    Loads a dataset and calculates additional properties
    Fields:
        n(int): Number of items
        m(int): Number of knapsack constraints
        b(ndarray(m,)): Knapsack bounds for each constraint
        a(ndarray(m,n)): Units of resources required if item i is taken
        c(ndarray(n,)): Units of profit if item i is taken
        u(ndarray(n,)): pseudo-utility of items
        seq(ndarray(n,)): item indices sorted by pseudo-utility in descending order
    '''
    
    
    def __init__(self, filename):
        self.load_file(filename)
        self.calculate_psuedo_utility()        
        
        
    def load_file(self, filename):
        with open(filename) as file:
            inp = []
            for line in file:
                for v in line.split():
                    inp.append(int(v))
            inp = np.array(inp)
            
            self.n = inp[0]
            self.m = inp[1]
            self.best_known = inp[2]
            
            assert len(inp) == (3 + self.n + self.m * self.n + self.m), 'Dataset is not valid!'
        
            inp = inp[3:]
            self.c = inp[:self.n]
            self.a = np.reshape(inp[self.n:self.n + self.n * self.m], (self.m, self.n))
            self.b = inp[self.n + self.n * self.m:]


    def calculate_psuedo_utility(self):
        f = np.concatenate((self.b, np.ones(self.n)))
        A = -np.concatenate((self.a.T, np.eye(self.n)), axis=1);
        b = -self.c;
        result = linprog(f, A, b)
        if result.success:
            shadow_price = result.x
            w = shadow_price[:self.m]
            self.u = self.c.T / (np.matmul(w.T, self.a))
            self.seq = (-self.u).argsort()
            # self.u[::-1].sort()
