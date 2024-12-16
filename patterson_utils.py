import numpy as np
from scipy.fft import fftn, ifftn
from scitbx.array_family import flex
from cctbx.development import random_structure
from cctbx import sgtbx
from cctbx import maptbx

def create_random_structure():
    # Create random structure
    n = np.random.randint(10, 30)
    elements = list(np.random.choice(['C', 'N', 'Cl', 'O'], size = n))

    xrs = random_structure.xray_structure(
            elements = elements,
            space_group_info = sgtbx.space_group_info(np.random.choice(['P21', 'C2'])),
            volume_per_atom = 20,
            random_u_iso=True
        )
    return xrs

def calculate_patterson_fft(Intensity, miller_indices, map_shape=(14, 14, 14), unit_cell_volume = 1.):
    """
    Calculate Patterson map using FFT method.
    
    Args:
        xrs: X-ray structure object
        d_min: Resolution limit (default: 0.8)
        map_shape: Base shape of the map (will be doubled) (default: (14,14,14))
    
    Returns:
        patterson_fft: Calculated Patterson map
        central_slice: Central slice of the Patterson map
    """
    # Get structure factor amplitudes and convert to numpy array
    #f_calc = xrs.structure_factors(d_min=d_min).f_calc()
    #f_obs = abs(f_calc).data().as_numpy_array()
    #miller_indices = np.array(f_calc.indices())

    # Create 3D grid for structure factors
    #map_shape = tuple([i*2 for i in map_shape])
    nx, ny, nz = map_shape
    F_grid = np.zeros((nx, ny, nz), dtype=np.complex128)

    # Place F^2 values on 3D grid at corresponding Miller indices
    for i, (h, k, l) in enumerate(miller_indices):
        F_squared = Intensity[i]
        
        # Generate equivalent reflections based on P21 symmetry
        equiv_hkls = get_equivalent_reflections(h, k, l)
        
        # Place F^2 at all equivalent positions
        for h_eq, k_eq, l_eq in equiv_hkls:
            # Convert indices to positive ones using modulo
            h_idx = h_eq % nx
            k_idx = k_eq % ny
            l_idx = l_eq % nz
            F_grid[h_idx, k_idx, l_idx] = F_squared

    # Calculate Patterson map using FFT
    patterson_fft = np.real(ifftn(F_grid)) / unit_cell_volume
    #patterson_fft = np.where(patterson_fft < 0, 0, patterson_fft)
    patterson_fft = (patterson_fft - patterson_fft.min()) / (patterson_fft.max() - patterson_fft.min())
    
    return patterson_fft

def get_equivalent_reflections(h, k, l):
    if k==0 and l==0:
        return [(h, 0, 0), (-h, 0, 0)]
    elif h==0 and l==0:
        return [(0, k, 0), (0, -k, 0)]
    elif h==0 and k==0:
        return [(0, 0, l), (0, 0, -l)]
    elif k == 0:
        return [(h, 0, l), (-h, 0, -l)]
    else:
        return [(h, k, l), (-h, k, -l), (-h, -k, -l), (h, -k, l)]

def recover_intensities_from_patterson(patterson_map, miller_indices, unit_cell_volume = 1.):
    """
    Recover structure factor intensities from Patterson map using FFT.
    
    Args:
        xrs: X-ray structure object
        patterson_map: 3D numpy array containing Patterson map
        miller_indices: Array of Miller indices (h,k,l)
    
    Returns:
        recovered_intensities: Array of recovered |F|² values
    """
    # Forward FFT of Patterson map
    F_recovered = fftn(patterson_map)
    
    # Get dimensions of Patterson map
    nx, ny, nz = patterson_map.shape
    
    # Initialize array for recovered intensities
    recovered_intensities = np.zeros(len(miller_indices))
    
    # Extract intensities at Miller indices positions
    for idx, (h, k, l) in enumerate(miller_indices):
        # Convert to positive indices using modulo
        h_idx = h % nx
        k_idx = k % ny
        l_idx = l % nz
        
        # Get the intensity value from the FFT grid
        recovered_intensities[idx] = np.abs(F_recovered[h_idx, k_idx, l_idx])
    
    # Normalize intensities
    recovered_intensities *= unit_cell_volume
    
    return recovered_intensities