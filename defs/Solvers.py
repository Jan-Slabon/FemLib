import numpy as np
from defs.logging.Log import Logging, Log_Level
from defs.Primitives import Function
class LinearEq():
    def __init__(self, A, f) -> None:
        self.A = A
        self.f = f
        Logging.Log(Log_Level.Info, "Initializing LinearEq")
    def solve_iter(self, epsilon) -> Function:
        x1 = np.random.randn(len(self.f))
        x0 = np.array(100*x1) # just to be sure that condition in a while loop is fulfilled
        while(np.linalg.norm(x1-x0)>= epsilon):
            x0 = np.array(x1)
            for i in range(len(self.f)):
                x1[i] = (self.f[i] - np.dot(self.A[i], x1) + self.A[i][i]*x1[i])/self.A[i][i]
        return x1

    def solve_exact(self) -> Function:
        return np.linalg.solve(self.A, self.f)

class LinearObstacleSolver():
    def __init__(self, A, b, obstacle) -> None:
        self.A = A
        self.b = b
        self.obstacle = obstacle
        Logging.Log(Log_Level.Info, "Initializing LinearObstacleSolver")

    def max(self, y, z):
        if y >= z:
            return y
        else: return z

    def solve(self, start, epsilon) -> np.ndarray: # Function
        x1 = start
        x0 = np.array(100*x1) # just to be sure that condition in a while loop is fulfilled
        k = 0
        while(np.linalg.norm(x1-x0)>= epsilon):
            k+=1
            x0 = np.array(x1)
            for i in range(len(self.b)):
                x1[i] = self.max((self.b[i] - np.dot(self.A[i], x1) + self.A[i][i]*x1[i])/self.A[i][i], self.obstacle[i])
            Logging.Log(Log_Level.Debug, "Error value after" + str(k) + "iteration: " + str(np.linalg.norm(x1-x0)))
        return x1

        