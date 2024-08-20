import numpy as np
def tensor_norm(tensor : np.array) -> np.array:
    norm = np.zeros((int(tensor.shape[0]/3)))
    stress_tensor = np.reshape(np.array(tensor), (3, int(tensor.shape[0]/3))).T
    for i,el in enumerate(stress_tensor):
        norm[i] = von_misses_norm(el)
    return norm
def cast_to_set(tensor : np.array, radius : int) -> np.array:
    stress_tensor = np.reshape(np.array(tensor), (3, int(tensor.shape[0]/3))).T # This way we are spliting stress components in 3 parts and are correctly matching vaues of stress tensor at given point x
    for i,el in enumerate(stress_tensor):
        norm = von_misses_norm(el)
        if norm > radius:
            stress_tensor[i] = radius * el/norm
    return np.reshape(stress_tensor, (tensor.shape[0]))

def von_misses_norm(vector : np.array) -> np.array:
    return np.sqrt(vector[0]**2 - vector[0]*vector[1] + vector[1]**2 + 3*vector[2]**2)