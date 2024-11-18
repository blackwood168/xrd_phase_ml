import torch
import numpy as np


def get_state_dict(weights_path):
    """Load model state dictionary from weights file."""
    state_dict = {}
    try:
        state_old = torch.load(weights_path)['state_dict']
        for key in state_old:
            key_new = key[6:]  # Remove 'model.' prefix
            state_dict[key_new] = state_old[key]
        print('Dictionary loaded successfully')
    except Exception as e:
        print(f"Error loading weights: {e}")
    return state_dict


# Define valid ranges for h,k,l indices for different Laue symmetry types
laue_types = {
    'romb': {'h': [0, 16], 'k': [0, 21], 'l': [0, 28]},
    'clin': {'h': [-13, 12], 'k': [0, 17], 'l': [0, 22]}, 
    'all': {'h': [-16, 16], 'k': [-14, 21], 'l': [0, 28]}
}


def make_dicts(laue_type):
    """Create mapping dictionaries between hkl indices and tensor positions."""
    hkl_minmax = laue_types[laue_type]
    dics = {'h': {}, 'k': {}, 'l': {}}
    dics_inv = {'h': {}, 'k': {}, 'l': {}}

    # Create forward and inverse mappings
    for letter in 'hkl':
        min_val, max_val = hkl_minmax[letter]
        for i, val in enumerate(range(min_val, max_val + 1)):
            dics[letter][val] = i
            dics_inv[letter][i] = val

    return (dics['h'], dics['k'], dics['l'],
            dics_inv['h'], dics_inv['k'], dics_inv['l'])


def create_intensity_tensors(intensity, ind_high, ind_low, laue_type,
                           h2ind, k2ind, l2ind):
    """Create intensity tensors for high and low resolution data."""
    hkl_minmax = laue_types[laue_type]
    num_high, num_low = 0, 0
    
    # Calculate tensor dimensions
    size = [1, 1]  # Batch and channel dimensions
    for letter in 'hkl':
        size.append(hkl_minmax[letter][1] - hkl_minmax[letter][0] + 1)
    
    low = np.zeros(size)
    high = np.zeros(size)

    # Fill high resolution tensor
    for j, (h, k, l) in enumerate(ind_high):
        if h in h2ind and k in k2ind and l in l2ind:
            high[0, 0, h2ind[h], k2ind[k], l2ind[l]] = intensity[j]
        else:
            num_high += 1

    # Fill low resolution tensor
    for h, k, l in ind_low:
        if h in h2ind and k in k2ind and l in l2ind:
            low[0, 0, h2ind[h], k2ind[k], l2ind[l]] = high[
                0, 0, h2ind[h], k2ind[k], l2ind[l]]
        else:
            num_low += 1

    assert low.shape == high.shape
    print(f'{num_high} / {len(ind_high)} indices did not fit into high tensor')
    print(f'{num_low} / {len(ind_low)} indices did not fit into low tensor')
    print(f'Tensor shapes: {low.shape} (low) and {high.shape} (high)')

    return torch.from_numpy(low).float(), torch.from_numpy(high).float()


def R_factor(out, y):
    """Calculate crystallographic R-factor between predicted and target intensities."""
    return torch.mean(
        torch.sum(torch.abs(torch.abs(out) - torch.abs(y)), dim=(1, 2, 3, 4)) /
        torch.sum(torch.abs(y), dim=(1, 2, 3, 4))
    )