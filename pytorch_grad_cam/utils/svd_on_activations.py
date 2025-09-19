import numpy as np
from sklearn.decomposition import KernelPCA
import torch

def get_2d_projection(activation_batch):
    # TBD: use pytorch batch svd implementation
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = (activations).reshape(
            activations.shape[0], -1).transpose()
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - \
            reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return np.float32(projections)


def get_2d_projection_tensor(activation_batch, device):
    if isinstance(activation_batch, np.ndarray):
        activations_tensor = torch.from_numpy(activation_batch).to(device)
    else:
        activations_tensor = activation_batch.to(device)
    activations_tensor[torch.isnan(activations_tensor)] = 0
    projections = []
    for activations in activations_tensor:
        reshaped_activations = activations.reshape(
            activations.shape[0], -1).transpose(0, 1)
        reshaped_activations = reshaped_activations - \
            reshaped_activations.mean(dim=0, keepdim=True)
        try:
            U, S, Vt = torch.linalg.svd(reshaped_activations, full_matrices=True)
        except AttributeError:
            U, S, Vt = torch.svd(reshaped_activations, some=False)
        projection = torch.matmul(reshaped_activations, Vt[0, :])
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    result = torch.stack(projections, dim=0)
    if isinstance(activation_batch, np.ndarray):
        return result.cpu().numpy().astype(np.float32)
    else:
        return result.float()


def get_2d_projection_kernel(activation_batch, kernel='sigmoid', gamma=None):
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = activations.reshape(activations.shape[0], -1).transpose()
        reshaped_activations = reshaped_activations - reshaped_activations.mean(axis=0)
        # Apply Kernel PCA
        kpca = KernelPCA(n_components=1, kernel=kernel, gamma=gamma)
        projection = kpca.fit_transform(reshaped_activations)
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return np.float32(projections)
