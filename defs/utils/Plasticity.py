import numpy as np
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
            stress_tensor[i] = hardening_radius * stress/norm
    return np.reshape(stress_tensor, (tensor.shape[0]))

def von_misses_norm(vector : np.array) -> np.array:
    return np.sqrt(vector[0]**2 - vector[0]*vector[1] + vector[1]**2 + 3*vector[2]**2)