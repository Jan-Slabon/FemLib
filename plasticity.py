from math import inf
import scipy
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import Delaunay
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from defs.Impl import Linear_Element, Linear_Map, Const_Linear_Element
from defs.Primitives import Point, Group, Operator, Function
from defs.Solvers import LinearObstacleSolver, IntervalObstacleSolver, Discriminator, TNNMG
from defs.utils.Plasticity import cast_to_set, tensor_norm, stress_offset, element_derivatives, pointvise_stress_norm
from matplotlib.animation import FuncAnimation
from defs.logging.Log import Logging, Log_Level
import time
def right_corner(x):
    if x[1] == e and x[0] >= 1.5 and x[0] <= 2:
        return -8
    elif x[1] == b and x[0] >= 0.5 and x[0] <= 0.7:
        return 0
    else:
        return 0
def right_side(x):
    if(x[0] == 2):
        return 0 #-10
    else : return 0
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
        self.sigma_size = len(grid)
        super().__init__(grid)
    def allocate(self) -> None:
        self.A = np.zeros((self.n, self.n))
        self.C = np.zeros((self.n, self.n))
        self.A2 = np.zeros((self.n, self.n))
        self.B = np.zeros((self.n*2, self.n*2))
        self.D = np.zeros((self.n*2))
        self.f1 = np.zeros(self.n)
        self.f2 = np.zeros(self.n)
        self.plasticity_first_part = np.zeros((self.n, self.sigma_size))
        self.plasticity_second_part = np.zeros((self.n, self.sigma_size))
        self.damage_field = np.zeros((self.extanded_boundary_size, self.extanded_boundary_size))
        self.value_product = np.zeros((self.extanded_boundary_size, self.extanded_boundary_size))

    def fill(self, triangle) -> None:
        element = Linear_Element([self.points[triangle[0]], self.points[triangle[1]], self.points[triangle[2]]])
        const_elements = Const_Linear_Element([self.points[triangle[0]], self.points[triangle[1]], self.points[triangle[2]]])
        desc = [self.enum[triangle[0]], self.enum[triangle[1]], self.enum[triangle[2]]]
        
        for (node1,enum) in zip(triangle,desc):
            if enum.group_id != Group.Dirichlet:
                const_lin_integrator = const_elements.build(self.points[node1])
                mixed_integral = const_lin_integrator.value_grad_integral()

                self.plasticity_first_part[enum.rid][self.id_elem] += mixed_integral[0]
                self.plasticity_second_part[enum.rid][self.id_elem] += mixed_integral[1]

            for (node2,enum2) in zip(triangle,desc):
                integrator = element.build(self.points[node1], self.points[node2])
                grad = integrator.gradient_integral()
                self.damage_field[enum2.id][enum.id] += grad
                val_integral = integrator.value_integral()
                self.value_product[enum2.id][enum.id] += val_integral

                if enum2.group_id != Group.Dirichlet and enum.group_id != Group.Dirichlet:

                    mix_grad = integrator.mixed_gradient_integral()

                    self.A[enum2.rid][enum.rid] += self.lamb * mix_grad[0][0] + self.mu * mix_grad[0][0] + 0.5 * self.mu * mix_grad[1][1]
                    self.C[enum2.rid][enum.rid] += self.lamb * mix_grad[1][0] + 0.5 * self.mu * mix_grad[0][1]
                    self.A2[enum2.rid][enum.rid] += self. lamb * mix_grad[1][1] + self.mu * mix_grad[1][1] + 0.5 * self.mu * mix_grad[0][0]


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
            h = lambda x : right_side(x)
            h2 = lambda x : right_corner(x)
            if len(neuman) > 1: # NonHomogenus Neuman
                for t, p in neuman:
                    for t2, p2 in neuman:
                        if p.id != p2.id:
                            self.f1[p.rid] += h(p.x)*0.5*np.linalg.norm(np.array(p.x) - np.array(p2.x))
                            self.f2[p.rid] += h2(p.x)*0.5*np.linalg.norm(np.array(p.x) - np.array(p2.x))
    def asemble_matrices(self) -> None:         # Fills large matrices with values of matrices calculated in fill()
        k = self.n
        k_extended = self.sigma_size
        self.M = np.zeros((2*k, 2*k))
        self.F = np.zeros(2*k)
        self.plasticity_A = np.zeros((k*2, k_extended*3))
        self.plasticity_B = np.zeros((k*2, k_extended*3))

        self.M[:k,:k] = self.A
        self.M[:k,k:] = self.C
        self.M[k:,:k] = self.C.T
        self.M[k:,k:] = self.A2

        self.F[:k] = self.f1
        self.F[k:] = self.f2

        self.plasticity_A[:k,:k_extended] = self.plasticity_first_part
        self.plasticity_A[k:,2*k_extended:] = self.plasticity_first_part

        self.plasticity_A[:k,2*k_extended:] = self.plasticity_second_part
        self.plasticity_A[k:,k_extended:2*k_extended] = self.plasticity_second_part

        self.plasticity_B[:k,:k_extended] = self.plasticity_first_part
        self.plasticity_B[:k,k_extended:2*k_extended] = self.plasticity_first_part

        self.plasticity_B[k:,:k_extended] = self.plasticity_second_part
        self.plasticity_B[k:,k_extended:2*k_extended] = self.plasticity_second_part
        
        self.Contact_Points = np.diag(self.B)
        self.Contact_Interval_Length = self.Contact_Points * 0.03
        self.Contact_Limits = np.array([ -1000 if x == 0 else -1000 for x in self.Contact_Points * 0.05])

    def assemble_damage_right_hside(self, stress_norm):
        multiplication_result = np.zeros(self.extanded_boundary_size)
        for (index, triangle) in enumerate(self.grid):
            desc = [self.enum[triangle[0]], self.enum[triangle[1]], self.enum[triangle[2]]]
            for (node, enum) in zip(triangle,desc):
                multiplication_result[enum.id] += Linear_Map(self.points[triangle[0]], self.points[triangle[1]], self.points[triangle[2]]).integral() * stress_norm[index]
        
        return multiplication_result
    
    def linear_to_constant_elements(self, values):
            const_projection = np.zeros(self.sigma_size)
            for (index, triangle) in enumerate(self.grid):
                a = self.points[triangle[0]]
                b = self.points[triangle[1]]
                c = self.points[triangle[2]]
                fa = values[triangle[0]]
                fb = values[triangle[1]]
                fc = values[triangle[2]]
                base_triangle_avg_value = (fa + fb + fc) / 12
                jacobian = Linear_Map(a,b,c).jacobian(0,0)
                const_projection[index] = base_triangle_avg_value * jacobian
            return const_projection

    def coulomb_friction_law(self, values):
        result = 0
        for triangle in self.grid:
            contact_points = [ (self.points[idx], idx) for idx in triangle if self.enum[idx].group_id == Group.Contact]
            edges = []
            if len(contact_points) <= 1:
                continue
            elif len(contact_points) == 2:
                edges.append((contact_points[0], contact_points[1]))
            elif len(contact_points) == 3:
                # TODO
                raise NotImplementedError()
            else:
                Logging.Log(Log_Level.Error, "Unsupported amount of vertex")
                raise NotImplementedError()
            for edge in edges:
                vert_l, id_l = edge[0]
                vert_r, id_r = edge[1]

                value_l_x = values[id_l]
                value_r_x = values[id_r]
                value_l_y = values[id_l + self.n]
                value_r_y = values[id_r + self.n]


                #tangential vector should be updated with regards to changed coordinates of vertices
                tangential_vector = (vert_r - vert_l) / np.linalg.norm(vert_r - vert_l)

                value_l = np.array([value_l_x, value_l_y]) @ tangential_vector
                value_r = np.array([value_r_x, value_r_y]) @ tangential_vector

                if value_l*value_r >= 0:
                    result += (abs(value_l) + abs(value_r)) * np.linalg.norm(vert_r - vert_l) / 2
                else:
                    f0 = abs(value_r) * np.linalg.norm(vert_r - vert_l) / (abs(value_l) + abs(value_r))
                    result += f0 * abs(value_l) / 2
                    result += (np.linalg.norm(vert_r - vert_l) - f0) * abs(value_r)
        return result
                    
    def prepare_tangential_and_size_of_boundary(self):
        tangential_vector = np.zeros((self.n,2))
        for triangle in self.grid:
            contact_points = [ (self.points[idx], idx) for idx in triangle if self.enum[idx].group_id == Group.Contact]
            edges = []
            if len(contact_points) <= 1:
                continue
            edges.append((contact_points[0], contact_points[1]))
            for edge in edges:
                vert_l, id_l = edge[0]
                vert_r, id_r = edge[1]

                non_normalize_tangential_vector = (vert_r - vert_l)
                tangential_vector[id_l] = non_normalize_tangential_vector
        return tangential_vector

    def solve(self): #-> List[Function]:
        self.asemble_matrices()
        innitial_F = np.array(self.F)
        k = self.n
        k_extended = self.sigma_size
        dt = 0.05        # Time step
        T = 1           # whole elapsed time of system
        friction_coef = 10 # friction coeficient
        viscosity = 1e3 #1e10   # viscosity parameter
        discriminator = Discriminator(self.enum, self.n)
        kappa = np.ones(k_extended)   # internal variable
        sigma = np.zeros(3*k_extended)   # stress
        sum_of_G = np.zeros(3*k_extended)
        u_0 = np.zeros(2*k)     # displacement
        damage = np.ones(self.extanded_boundary_size)
        dispalcement_history = []
        stress_history = []
        kappa_history = []
        damage_history = []
        start = time.time()
        tangential_vector = self.prepare_tangential_and_size_of_boundary()
        tnnmg_solver : TNNMG = TNNMG(self.M, self.F -  dt * self.plasticity_A @ sum_of_G, friction_coef * np.reshape(tangential_vector, self.n * 2), self.Contact_Limits, discriminator)

        for i in range(int(T/dt)):
            print("Iteration", str(i), "out of ", str(int(T/dt - 1)))
            dispalcement_history.append(np.array(u_0))
            damage_history.append(np.array(damage))
            kappa_history.append(kappa)
            stress_history.append(tensor_norm(sigma))

            if i <= int( int(T/dt) / 2):
                self.F = 2* innitial_F * i* dt
            elif i <= int(T/dt):
                self.F = 2* innitial_F * int( int(T/dt) / 2) * dt - 2* innitial_F * (i - int( int(T/dt) / 2))* dt
            else:
                self.F = np.zeros(innitial_F.shape)
            if i <= int( int(T/dt) / 2):
                self.F = 2* innitial_F * i* dt
            elif i <= int(T/dt):
                self.F = 2* innitial_F * int( int(T/dt) / 2) * dt - 2* innitial_F * (i - int( int(T/dt) / 2))* dt
            else:
                self.F = np.zeros(innitial_F.shape)

            # Div(sigma) = f
            #res = minimize(fun = lambda x: 0.5 * x @ self.M @ x - x @ self.F + dt * x @ self.plasticity_A @ sum_of_G + friction_coef * np.sum(tangential_vector * np.reshape(x, (self.n,2))), x0 = u_0, method="Powell")
            #u = np.linalg.solve(self.M, self.F -  dt * self.plasticity_A @ sum_of_G) # + friction_coef * np.reshape(tangential_vector, (2 * self.n)))
            tnnmg_solver.update_rhs(self.F -  dt * self.plasticity_A @ sum_of_G)
            u = tnnmg_solver.solve()
            # sigma = Ce(u) + dt*G(sigma) - Ce(u_0) + sigma
            sigma_diference = sigma - cast_to_set(sigma, 430 * kappa* self.linear_to_constant_elements(damage)) # 430

            u1_dx_dy = element_derivatives(self.grid, self.enum, u[:k])
            u2_dx_dy = element_derivatives(self.grid, self.enum, u[k:])


            sum_of_G[:k_extended] += (-1/(2*viscosity) * (self.lamb *(sigma_diference[:k_extended] + sigma_diference[k_extended:2*k_extended]) +  self.mu * sigma_diference[:k_extended]))
            sum_of_G[k_extended:2*k_extended] += (-1/(2*viscosity) * (self.lamb *(sigma_diference[:k_extended] + sigma_diference[k_extended:2*k_extended]) +  self.mu * sigma_diference[k_extended:2*k_extended]))
            sum_of_G[2*k_extended:] +=  (self.mu * (-1/(2*viscosity)) * sigma_diference[2*k_extended:])

            sigma[:k_extended] = (self.lamb * (u1_dx_dy[:,0] + u2_dx_dy[:, 1]) +  self.mu * u1_dx_dy[:,0])
            sigma[k_extended:2*k_extended] = (self.lamb* (u1_dx_dy[:,0] + u2_dx_dy[:, 1]) +  self.mu * u2_dx_dy[:, 1])
            sigma[2*k_extended:] = (self.lamb* 0.5 * (u2_dx_dy[:, 0] + u1_dx_dy[:,1]))
            
            sigma += dt * sum_of_G
            # kappa = 1/2*lambda * ||sigma - P(sigma)||
            kappa += dt * (1/(2*viscosity)) * stress_offset(sigma, kappa)
            # damage' = div(grad(damage)) + f
            linear_obstacle_solver = IntervalObstacleSolver(self.value_product + 0.0001 * dt * self.damage_field, self.value_product @ damage  - dt * self.assemble_damage_right_hside(  1/100 *  pointvise_stress_norm(sigma)))
            damage = linear_obstacle_solver.solve(np.ones(damage.shape), 1e-3)

            u_0 = u

        end = time.time()
        print("Elapsed time", end - start)
        return dispalcement_history, stress_history, kappa_history, damage_history


n = 30
p1=0; k = 2; b = 0; e = 1
n = 30
p1=0; k = 2; b = 0; e = 1
dx = (k - p1) / n
dy = (b - e) / int(n/3)
truncated_shape = (int(n/3), n-1)
shape = (int(n/3), n)
time_steps = 20
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
    elif p[1] == b:
        enum.append(Point(p, i, r, Group.Contact))
        r+=1
    elif p[1] == e or p[0] == k:
        enum.append(Point(p, i, r, Group.Neuman))
        r+=1
    else:
        enum.append(Point(p, i, r, Group.Interior))
        r+=1
    i+=1
    
tri = Delaunay(points)
i = 0
j = 0
opearator = Linear_Operator(tri.simplices, points, enum, r, 1098, 769, dx, dy, truncated_shape, shape) # 80769, 121153 / 1098, 769
opearator.construct()
dispacement, stress, internal_var, damage = opearator.solve()

fig, [ax1, ax2] = plt.subplots(1,2)
def animate(i):
    sig = stress[i % len(stress)]
    plt.suptitle("T = " + str( '%.1f'%(i*0.05) ))
    ax1.clear()
    ax1.set_title("Stress")
    c = ax1.tripcolor(points[:,0], points[:,1], sig, triangles = tri.simplices)
    kapp = internal_var[i % len(internal_var)]
    kapp = internal_var[i % len(internal_var)]
    ax2.clear()
    ax2.set_title("Internal Variable")
    ax2.tripcolor(points[:,0], points[:,1], kapp, triangles = tri.simplices)

fig3 = plt.figure()
ax3 = fig3.add_subplot(projection='3d')
def animate_damage(i):
    dam = damage[i % len(damage)]
    ax3.clear()
    ax3.set_title("Damage")
    ax3.set_zlim(0,1)
    ax3.plot_trisurf(points[:,0], points[:,1], dam, triangles = tri.simplices, linewidth=0.2, antialiased=True)
    #ax3.imshow(dam.reshape(truncated_shape))

ani = FuncAnimation(
    fig, animate, time_steps, interval=100)
ani.save("stress_2.gif")

ani3 = FuncAnimation(
    fig3, animate_damage, time_steps, interval=100)
ani3.save("damage_2.gif")
fig2, ax = plt.subplots(1,1)
def animate_dispacement(i):
    global points
    u_t = dispacement[i % len(dispacement)]
    plt.suptitle("T = " + str( '%.1f'%(i*0.05) ))
    ax.clear()
    plt.ylim(b - 0.1*(e - b), e + 0.1*(e - b))
    plt.xlim(p1 - 0.1*(k - p1), k + 0.1*(k - p1))
    plt.ylim(b - 0.1*(e - b), e + 0.1*(e - b))
    plt.xlim(p1 - 0.1*(k - p1), k + 0.1*(k - p1))
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
    fig2, animate_dispacement, time_steps, interval=100)
ani2.save("displacement_3.gif")

u_t = 3* dispacement[-1]
ax.clear()
plt.ylim(-0.3, e + 0.1*(e - b))
plt.xlim(0, k + 0.1*(k - p1))
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
plt.savefig("final_displacement_3.png")


fig, [ax1, ax2] = plt.subplots(1,2)
sig = stress[-1]
plt.suptitle("T = " + str( 1 ))
ax1.clear()
ax1.set_title("Stress")
c = ax1.tripcolor(points[:,0], points[:,1], sig, triangles = tri.simplices)
kapp = internal_var[i % len(internal_var)]
ax2.clear()
ax2.set_title("Internal Variable")
ax2.tripcolor(points[:,0], points[:,1], kapp, triangles = tri.simplices)
fig.savefig("final_stress_internalvar_2.png")