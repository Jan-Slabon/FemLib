import numpy as np
from defs.Primitives import Group
def tensor_norm(tensor : np.array) -> np.array:
    norm = np.zeros((int(tensor.shape[0]/3)))
    stress_tensor = np.reshape(np.array(tensor), (3, int(tensor.shape[0]/3))).T
    for i,el in enumerate(stress_tensor):
        norm[i] = von_misses_norm(el)
    return norm
def stress_offset(tensor : np.array, kappa : np.array) -> np.array:
    return pointvise_stress_norm(tensor - cast_to_set(tensor, kappa))
def pointvise_stress_norm(tensor : np.array) -> np.array:
    stress_tensor = np.reshape(np.array(tensor), (3, int(tensor.shape[0]/3))).T # This way we are spliting stress components in 3 parts and are correctly matching vaues of stress tensor at given point x
    offset = np.zeros(int(tensor.shape[0]/3))
    for i,el in enumerate(stress_tensor):
        norm = np.linalg.norm(el)
        offset[i] = norm
    return offset
def cast_to_set(tensor : np.array, kappa : np.array) -> np.array:
    stress_tensor = np.reshape(np.array(tensor), (3, int(tensor.shape[0]/3))).T # This way we are spliting stress components in 3 parts and are correctly matching vaues of stress tensor at given point x
    for i,(stress, hardening_radius) in enumerate(zip(stress_tensor, kappa)):
        norm = von_misses_norm(stress)
        if norm > hardening_radius:
            stress_tensor[i] = hardening_radius * (stress/norm)
    return np.reshape(stress_tensor.T, (tensor.shape[0]))

def von_misses_norm(vector : np.array) -> np.array:
    return np.sqrt(vector[0]**2 - vector[0]*vector[1] + vector[1]**2 + 3*vector[2]**2)

def truncate_shape(tensor, enum):
    out_tensor = []
    for (i, el) in enumerate(tensor):
        if enum[i % len(enum)].group_id != Group.Dirichlet:
            out_tensor.append(el)
    return np.array(out_tensor)

def plane_derivatives(a, b, c, fa, fb, fc):
    A = np.concat([a,[fa]])
    B = np.concat([b,[fb]])
    C = np.concat([c, [fc]])
    normal = np.cross((B-A), (C-A))
    return np.array([-normal[0]/normal[2], -normal[1]/normal[2]])

def element_derivatives(mesh, point_data, values):
    derivatives = []
    for triangle in mesh:
        a,b,c = point_data[triangle[0]], point_data[triangle[1]], point_data[triangle[2]]
        if a.group_id == Group.Dirichlet:
            fa = 0
        else:
            fa = values[a.rid]

        if b.group_id == Group.Dirichlet:
            fb = 0
        else:
            fb = values[b.rid]

        if c.group_id == Group.Dirichlet:
            fc = 0
        else:
            fc = values[c.rid]

        dx, dy = plane_derivatives(a.x, b.x, c.x, fa, fb, fc)
        derivatives.append([dx,dy])
    return np.array(derivatives)
        