import numpy as np

class LinearEq():
    def __init__(self, A, f) -> None:
        self.A = A
        self.f = f
    
    def solve_iter(self, epsilon):
        x1 = np.random.randn(len(self.f))
        x0 = np.array(100*x1) # just to be sure that condition in a while is fulfilled
        while(np.linalg.norm(x1-x0)>= epsilon):
            x0 = np.array(x1)
            for i in range(len(self.f)):
                x1[i] = (self.f[i] - np.dot(self.A[i], x1) + self.A[i][i]*x1[i])/self.A[i][i]
        return x1
    def solve_exact(self):
        return np.linalg.solve(self.A, self.f)