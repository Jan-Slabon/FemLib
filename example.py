from math import inf
import scipy
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import Delaunay
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from defs.Impl import Linear_Element, Linear_Map
from defs.Primitives import Point, Group, Operator, Function
from defs.Solvers import LinearObstacleSolver
from defs.utils.Plasticity import cast_to_set, tensor_norm, stress_offset, truncate_shape
from matplotlib.animation import FuncAnimation
import time
def right_corner(x):
    if x[0]<= 0.5 and x[0]>= 0.2:
        if x[1] == e:
            return -30
        if x[1] == b:
            return 30
    else:
        return 0
def is_zero(arr):
    flag = True
    for array_el in arr:
        if array_el != 0.0:
            print("nie jest rowne!")
            flag = False
            break
    return flag
class Linear_Operator(Operator):
    def __init__(self, grid, points, enum, n, a, b, dx, dy, truncated_grid_shape, grid_shape) -> None:
        self.points = points
        self.enum = enum
        self.n = n
        self.lamb= a
        self.mu= b
        self.dx = dx
        self.dy = dy
        self.truncated_shape = truncated_grid_shape
        self.shape = grid_shape
        self.extanded_boundary_size = grid_shape[0] * grid_shape[1]
        super().__init__(grid)
    def allocate(self) -> None:
        self.A = np.zeros((self.n, self.n))
        self.C = np.zeros((self.n, self.n))
        self.A2 = np.zeros((self.n, self.n))
        self.B = np.zeros((self.n*2, self.n*2))
        self.D = np.zeros((self.n*2))
        self.f1 = np.zeros(self.n)
        self.f2 = np.zeros(self.n)
        self.plasticity_first_part = np.zeros((self.extanded_boundary_size, self.extanded_boundary_size))
        self.plasticity_second_part = np.zeros((self.extanded_boundary_size, self.extanded_boundary_size))

    def fill(self, triangle) -> None:
        element = Linear_Element([self.points[triangle[0]], self.points[triangle[1]], self.points[triangle[2]]])
        desc = [self.enum[triangle[0]], self.enum[triangle[1]], self.enum[triangle[2]]]
        
        for (node1,enum) in zip(triangle,desc):
            for (node2,enum2) in zip(triangle,desc):

                integrator = element.build(self.points[node1], self.points[node2])
                grad = integrator.gradient_integral()
                mix_grad = integrator.mixed_gradient_integral()
                mixed_integral = integrator.mixed_value_grad_integral()

                self.plasticity_first_part[enum.rid][enum2.rid] += mixed_integral[0]
                self.plasticity_second_part[enum.rid][enum2.rid] += mixed_integral[2]

                if enum.group_id == Group.Dirichlet or enum2.group_id == Group.Dirichlet:
                    continue

                self.A[enum.rid][enum2.rid] += self.lamb * mix_grad[0][0] + 2 * self.mu * mix_grad[0][0] + self.mu * mix_grad[1][1]
                self.C[enum.rid][enum2.rid] += self.lamb * mix_grad[0][1] + self.mu * mix_grad[1][0]
                self.A2[enum.rid][enum2.rid] += self. lamb * mix_grad[1][1] + 2 * self.mu * mix_grad[1][1] + self.mu * mix_grad[0][0]


                if enum.group_id == Group.Contact:
                    self.B[enum.rid+self.n][enum.rid+self.n] = -1
                    self.D[enum.rid+self.n] = 0.05
            tr = []
            tr2 = []
            for i in range(3):
                if self.enum[triangle[i]].group_id != Group.Dirichlet:
                    tr.append(Linear_Map(self.points[triangle[i%3]], self.points[triangle[(i+1)%3]], self.points[triangle[(i+2)%3]]))
                    tr2.append(self.enum[triangle[i]])
            neuman = []
            for trian, p in zip(tr,tr2):
                if p.group_id == Group.Interior:
                    pass
                    # area, err = integrate.dblquad(lambda x,y : trian.shape_function(x,y) * function(trian.map_triangle([x,y]))[0] *
                    # trian.jacobian(x,y), 0, 1, lambda x : 0, lambda x : 1 - x, epsabs=1.5e-4, epsrel=1.5e-4)
                    # area2, err2 = integrate.dblquad(lambda x,y : trian.shape_function(x,y) * function(trian.map_triangle([x,y]))[1] *
                    # trian.jacobian(x,y), 0, 1, lambda x : 0, lambda x : 1 - x, epsabs=1.5e-4, epsrel=1.5e-4)
                    # self.f1[p.rid] += area
                    # self.f2[p.rid] += area2
                elif p.group_id == Group.Neuman:
                    neuman.append([trian,p])
            h = lambda x : 0
            h2 = lambda x : right_corner(x)
            if len(neuman) > 1: # NonHomogenus Neuman
                for t, p in neuman:
                    for t2, p2 in neuman:
                        if p.id != p2.id:
                            self.f1[p.rid] += h(p.x)*0.5*np.linalg.norm(np.array(p.x) - np.array(p2.x))
                            self.f2[p.rid] += h2(p.x)*0.5*np.linalg.norm(np.array(p.x) - np.array(p2.x))
    def asemble_matrices(self) -> None:         # Fills large matrices with values of matrices calculated in fill()
        k = self.n
        k_extended = self.extanded_boundary_size
        self.M = np.zeros((2*k, 2*k))
        self.F = np.zeros(2*k)
        self.plasticity_A = np.zeros((k_extended*2, k_extended*3))
        self.plasticity_B = np.zeros((k_extended*2, k_extended*3))

        self.M[:k,:k] = self.A
        self.M[:k,k:] = self.C
        self.M[k:,:k] = self.C.T
        self.M[k:,k:] = self.A2

        self.F[:k] = self.f1
        self.F[k:] = self.f2

        self.plasticity_A[:k_extended,:k_extended] = self.plasticity_first_part
        self.plasticity_A[k_extended:,2*k_extended:] = self.plasticity_first_part

        self.plasticity_A[:k_extended,2*k_extended:] = self.plasticity_second_part
        self.plasticity_A[k_extended:,k_extended:2*k_extended] = self.plasticity_second_part

        self.plasticity_B[:k_extended,:k_extended] = self.plasticity_first_part
        self.plasticity_B[:k_extended,k_extended:2*k_extended] = self.plasticity_first_part

        self.plasticity_B[k_extended:,:k_extended] = self.plasticity_second_part
        self.plasticity_B[k_extended:,k_extended:2*k_extended] = self.plasticity_second_part
        
        self.Contact_Points = np.diag(self.B)
        self.Contact_Interval_Length = self.Contact_Points * 0.03
        self.Contact_Limits = np.array([ -1000 if x == 0 else x for x in self.Contact_Points * 0.05])

    def solve(self): #-> List[Function]:
        self.asemble_matrices()

        k = self.n
        k_extended = self.extanded_boundary_size
        dt = 0.1        # Time step
        T = 1           # whole elapsed time of system
        viscosity = 1e13   # viscosity parameter
        kappa = np.ones(k_extended)   # internal variable
        sigma = np.zeros(3*k_extended)   # stress
        sum_of_G = np.zeros(3*k_extended)
        u_0 = np.zeros(2*k)     # displacement

        dispalcement_history = []
        stress_history = []
        kappa_history = []
        start = time.time()


        for i in range(int(T/dt)):
            dispalcement_history.append(np.array(u_0))
            stress_history.append(tensor_norm(sigma))
            kappa_history.append(np.array(kappa))

            u = np.linalg.solve(self.M, self.F + self.M @ u_0 - truncate_shape(self.plasticity_A @ sigma, self.enum) + dt * (1/(2 * viscosity)) * truncate_shape((2 * self.mu * self.plasticity_A + self.lamb * self.plasticity_B) @ (sigma - cast_to_set(sigma, 430 * kappa)), self.enum))
            
            sigma_diference = sigma - cast_to_set(sigma, 430 * kappa)
            # kolejnosc pochodnych dy, dx jest podejrzana
            u_extended = np.zeros(self.shape)
            u_extended[:,1:] = u[:k].reshape(self.truncated_shape)
            grad_ux_dy, grad_ux_dx     = np.gradient(u_extended, self.dy, self.dx)
            grad_ux_dy, grad_ux_dx     = grad_ux_dx.reshape((k_extended)), grad_ux_dy.reshape((k_extended))

            u_extended = np.zeros(self.shape)
            u_extended[:,1:] = u[k:].reshape(self.truncated_shape)
            grad_uy_dy, grad_uy_dx     = np.gradient(u_extended, self.dy, self.dx)
            grad_uy_dy, grad_uy_dx     = grad_uy_dx.reshape((k_extended)), grad_uy_dy.reshape((k_extended))

            u_extended = np.zeros(self.shape)
            u_extended[:,1:] = u_0[:k].reshape(self.truncated_shape)
            grad_u0x_dy, grad_u0x_dx    = np.gradient(u_extended,self.dy, self.dx)
            grad_u0x_dy, grad_u0x_dx    = grad_u0x_dx.reshape((k_extended)), grad_u0x_dy.reshape((k_extended))

            u_extended = np.zeros(self.shape)
            u_extended[:,1:] = u_0[k:].reshape(self.truncated_shape)
            grad_u0y_dy , grad_u0y_dx   = np.gradient(u_extended, self.dy, self.dx)
            grad_u0y_dy , grad_u0y_dx   = grad_u0y_dx.reshape((k_extended)), grad_u0y_dy.reshape((k_extended))



            sigma[:k_extended] -= dt*(1/(2*viscosity) * (self.lamb *(sigma_diference[:k_extended] + sigma_diference[k_extended:2*k_extended]) + 2 * self.mu * sigma_diference[:k_extended]))
            sigma[k_extended:2*k_extended] -= dt*(1/(2*viscosity) * (self.lamb *(sigma_diference[:k_extended] + sigma_diference[k_extended:2*k_extended]) + 2 * self.mu * sigma_diference[k_extended:2*k_extended]))
            sigma[2*k_extended:] -= dt*(self.mu * (1/viscosity) * sigma_diference[2*k_extended:])

            sigma[:k_extended] += (self.lamb * (grad_ux_dx + grad_uy_dy) + 2 * self.mu * grad_ux_dx)
            sigma[k_extended:2*k_extended] += (self.lamb* (grad_ux_dx + grad_uy_dy) + 2 * self.mu * grad_uy_dy)
            sigma[2*k_extended:] += (self.lamb* (grad_uy_dx + grad_ux_dy))

            sigma[:k_extended] -= self.lamb* (grad_u0x_dx + grad_u0y_dy) + 2 * self.mu * grad_u0x_dx
            sigma[k_extended:2*k_extended] -= self.lamb* (grad_u0x_dx + grad_u0y_dy) + 2 * self.mu * grad_u0y_dy
            sigma[2*k_extended:] -= self.lamb * (grad_u0y_dx + grad_u0x_dy)
            # sigma = Ce(u) + dt*G(sigma) - Ce(u_0) + sigma
            kappa += dt * (1/(2*viscosity)) * stress_offset(sigma, kappa)
            # kappa = 1/2*lambda * ||sigma - P(sigma)||
            u_0 = u

        end = time.time()
        print("Elapsed time", end - start)
        return dispalcement_history, stress_history, kappa_history

n = 36
p1=0; k = 0.8; b = 0; e = 1/4
dx = (k - p1) / n
dy = (b - e) / int(n/3)
truncated_shape = (int(n/3), n-1)
shape = (int(n/3), n)
x = np.linspace(p1,k,n)
y = np.linspace(b,e,int(n/3))
pointsx, pointsy = np.meshgrid(x,y)
vert = []
for i, j in zip(pointsx, pointsy):
    for a,c in zip(i,j):
        vert.append([a,c])

points = np.array(vert)
enum = []
i=0
r=0
for p in points:
    if p[0] == p1:
        enum.append(Point(p, i, -1, Group.Dirichlet))
    elif p[1] == e or p[1] == b:
        enum.append(Point(p, i, r, Group.Neuman))
        r+=1
    elif p[1] == b and p[0]>=0.3 and p[0]<=0.5:
        enum.append(Point(p, i, r, Group.Contact))
        r+=1
    else:
        enum.append(Point(p, i, r, Group.Interior))
        r+=1
    i+=1
    
tri = Delaunay(points)
i = 0
j = 0
opearator = Linear_Operator(tri.simplices, points, enum, r, 80769, 121153, dx, dy, truncated_shape, shape)
opearator.construct()
dispacement, stress, internal_var = opearator.solve()

fig, [ax1, ax2] = plt.subplots(1,2)
def animate(i):
    sig = stress[i % len(stress)]
    plt.suptitle("T = " + str( '%.1f'%(i*0.1) ))
    ax1.clear()
    ax1.set_title("Stress")
    c = ax1.imshow(sig.reshape(shape), origin='lower',interpolation='bilinear')
    kapp = internal_var[i]
    ax2.clear()
    ax2.set_title("Internal Variable")
    ax2.imshow(kapp.reshape(shape), origin='lower',interpolation='bilinear')

ani = FuncAnimation(
    fig, animate, 10, interval=500)
ani.save("stress.gif")


fig2, ax = plt.subplots(1,1)
def animate_dispacement(i):
    global points
    u_t = dispacement[i % len(dispacement)]
    ax.clear()
    plt.ylim(-0.2, 0.5)
    plt.xlim(-0.1, 0.9)
    ax.set_title("Displacement")
    l=0
    pointsz = np.array(points)
    res = np.reshape(u_t, (2,r))
    res = np.transpose(res)
    for iter in range(np.shape(pointsz)[0]):
        if pointsz[iter][0] == p1:
            pass
        else:
            pointsz[iter]+=res[l]
            l+=1
    points = np.transpose(points)
    pointsz = np.transpose(pointsz)
    ax.triplot(pointsz[0],pointsz[1], tri.simplices, color='blue')
    points = np.transpose(points)

ani2 = FuncAnimation(
    fig2, animate_dispacement, 10, interval=500)
ani2.save("displacement.gif")