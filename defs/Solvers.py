import numpy as np
from defs.logging.Log import Logging, Log_Level
from defs.Primitives import Function, Group
def print_matrix(M):
    for row in M:
        for col in row:
            print(col, end=" ")
        print()
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
    
class IntervalObstacleSolver():
    def __init__(self, A, b):
        self.A = A
        self.b = b
    def solve(self, start, epsilon):
        x1 = start
        x0 = np.array(100*x1 + 1) # just to be sure that condition in a while loop is fulfilled
        k = 0
        while(np.linalg.norm(x1-x0)>= epsilon):
            k+=1
            x0 = np.array(x1)
            for i in range(len(self.b)):
                x1[i] = np.clip((self.b[i] - np.dot(self.A[i], x1) + self.A[i][i]*x1[i])/self.A[i][i], 0,1)
            Logging.Log(Log_Level.Debug, "Error value after " + str(k) + " iteration: " + str(np.linalg.norm(x1-x0)))
        return x1

class Discriminator:
        def __init__(self, enum, mesh_size):
            self.contact_boundary = np.zeros(mesh_size)
            self.n = mesh_size
            for i,enumeration in enumerate(enum):
                if(enumeration.group_id == Group.Contact):
                    self.contact_boundary[enumeration.rid] = 1
        # calculates if two floating point nomber are close to each other
        def close_to(self, floating_number_1, floating_number_2):
            # epsilon could be case dependent to accelerate convergence
            epsilon = 0.001
            if(abs(floating_number_1-floating_number_2) <= epsilon):
                return True
            else: return False

        def get_non_diff_subspaces(self, friction):
            result = np.zeros(len(friction) * 2) # friction here is defined as per node in triangulation and not per subspace
            num_of_non_diff_subspaces = 0
            for i,el in enumerate(friction):
                if(self.contact_boundary[i] == 1):
                    if(self.close_to(abs(el[0] + el[1]),0)):
                        num_of_non_diff_subspaces += 2
                        result[i] = 1
                        result[i + self.n] = 1
            return result, num_of_non_diff_subspaces
        def diff_helper(self, friction):
            result = np.zeros(len(friction) * 2) # friction here is defined as per node of triangulation and not per subspace
            for i,el in enumerate(friction):
                if(self.contact_boundary[i] == 1):
                    if(el[0] + el[1] >= 0):
                        result[i] = 1
                        result[i + self.n] = 1
                    else:
                        result[i] = -1
                        result[i + self.n] = -1
            return result

class TNNMG:
    def __init__(self, A, b, j, D, discriminator: Discriminator):
        self.A = A
        self.b = b
        self.j = j
        self.Dj = np.diag(j)
        self.Domain = D
        self.L = np.array(A)
        self.D = np.diag(A)
        self.R = np.array(A)
        self.discriminator = discriminator
        for i in range(1, len(A) + 1):
            self.L[i - 1][i - 1:] = 0
            self.R[i - 1][:i] = 0
    def hard_truncate_matrix(self, A, ind, num):
        truncated_shape = (A.shape[0] - num, A.shape[0] - num)
        truncated = np.zeros(truncated_shape)
        current_row = 0
        for i, row in enumerate(A):
            if ind[i] == 1:
                continue
            current_col = 0
            for j, col in enumerate(row):
                if ind[j] == 1:
                    continue
                else:
                    truncated[current_row][current_col] = col
                    current_col+=1
            current_row+=1
        return truncated
    def hard_truncate_vector(self, v, ind, num):
        truncated_shape = v.shape[0] - num
        truncated_vector = np.zeros(truncated_shape)
        current_index = 0
        for i in range(v.shape[0]):
            if ind[i] == 1:
                continue
            else:
                truncated_vector[current_index] = v[i]
                current_index+=1
        return truncated_vector
    def expand_vector(self, orig, vec, ind):
        extended_vector = np.zeros(orig.shape)
        row = 0
        for i in range(extended_vector.shape[0]):
            if ind[i] == 1:
                extended_vector[i] = orig[i]
            else:
                extended_vector[i] = vec[row]
                row += 1
        return extended_vector
    def truncate_matrix(self, A, ind):
            A_t = np.array(A)
            for i, el in enumerate(ind):
                if el != 0:
                    A_t[i][:] = 0
                    A_t[:][i] = 0
            return A_t
    def truncate_vector(self, v, ind):
        v_t = np.array(v)
        for i, el in enumerate(ind):
                if el != 0:
                    v_t[i] = 0
        return v_t
    def point_norm(self, vector):
        res = np.zeros(self.discriminator.n)
        for i, el in enumerate(vector):
            res[i] = abs(el[0] + el[1]) # its asumed that vector has shape (n, 2)
        return res
    def gauss_seidl_step(self, u):
        friction = np.reshape(self.j * u, (int(len(u) / 2), 2))
        j_sign = self.discriminator.diff_helper(friction)
        x = self.gauss_seidl_step_clean(u, self.A + j_sign * self.Dj, self.b)
        return x
    def gauss_seidl_step_clean(self, u, A, b):
        x = np.array(u)
        for i in range(len(u)):
            x[i] = max((b[i] - np.dot(A[i], x) + A[i][i]*x[i])/A[i][i], self.Domain[i])
        return x
    def truncated_newton_step(self, u):
        friction = np.reshape(self.j * u, (int(len(u) / 2), 2))
        j_sign = self.discriminator.diff_helper(friction)
        non_diff_ind, number_of_errased_rows = self.discriminator.get_non_diff_subspaces(friction)
        
        Hessian_t = self.hard_truncate_matrix(self.A, non_diff_ind, number_of_errased_rows)
        Jacobian_t = self.hard_truncate_vector(self.A @ u - self.b - self.j * j_sign, non_diff_ind, number_of_errased_rows)
        x_t = np.linalg.solve(Hessian_t, -Jacobian_t)
        x_t = x_t + self.hard_truncate_vector(u, non_diff_ind, number_of_errased_rows)
        x = self.expand_vector(u, x_t, non_diff_ind)
        # Hessian_t = self.truncate_matrix(self.A, non_diff_ind)
        # Jacobian_t = self.truncate_vector(self.A @ u - self.b - self.j * j_sign, non_diff_ind)
        # x = u - np.linalg.pinv(Hessian_t) @ Jacobian_t
        return x
    def non_smooth_newton_step(self, u):
        friction = np.reshape(self.j * u, (int(len(u) / 2), 2))
        j_sign = self.discriminator.diff_helper(friction)

        Hessian_t = self.A
        Jacobian_t = self.A @ u - self.b - self.j * j_sign
        x = np.linalg.solve(Hessian_t, -Jacobian_t)
        x = x + u
        return x
    def projection_step(self, u):
        u_projected = np.array(u)
        for i, el in enumerate(self.Domain):
            if el < -1000:
                continue
            else:
                if u[i] < el:
                    u_projected[i] = el
        return u_projected

    def linear_search_step(self, u0, u1):
        difference_len = np.linalg.norm(u1 - u0)
        if difference_len == 0.0:
            return u0
        direction = (u1 - u0) / difference_len
        step_size = difference_len / 10
        functional = lambda u : 1/2 * u @ self.A @ u - self.b @ u + np.sum(self.point_norm(np.reshape(self.j * u, (int(len(u)/2), 2))))
        min_val = functional(u0)
        min_point = u0
        for x in range(1, 10 + 1):
            curr_point = u0 + direction * step_size * x
            curr_val = functional(curr_point)
            if(curr_val < min_val):
                min_val = curr_val
                min_point = curr_point
        return min_point
    def update_rhs(self, rhs):
        self.b = rhs
    def step(self, u_start):
        u_1 = self.gauss_seidl_step(u_start)
        #u_2 = self.truncated_newton_step(u_1)
        u_2 = self.non_smooth_newton_step(u_1)
        u_3 = self.projection_step(u_2)
        u_4 = self.linear_search_step(u_1, u_3)
        return u_4

    def solve(self):
        epsilon = 1e-3
        x0 = np.absolute(np.random.normal(0, 1, self.b.shape))
        x1 = self.step(x0)
        while(np.linalg.norm(x0 - x1) > epsilon):
            x0 = x1
            x1 = self.step(x0)
        return x1
        


        