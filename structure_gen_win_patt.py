import core
import utils
from utils import GenBuilder, distr_gen, sample_gen

import numpy as np
import scipy.stats as sts
import itertools as it
import multiprocessing as mp
from multiprocessing import freeze_support
import pandas as pd
from cctbx import crystal, miller, uctbx, sgtbx, xray
from cctbx.array_family import flex
import patterson_utils as pu

# Structure space groups for different databases/systems
#SPACE_GROUPS = ['P212121', 'P21', 'C2', 'P21212', 'C2221', 'P1'] #PDB
#SPACE_GROUPS = ['P21/c', "P-1", "P21", 'P212121', "C2/c", "Pbca"] #CSD
#SPACE_GROUPS = ['C2221', 'P21212', 'P212121'] #orthoromb
SPACE_GROUPS = ['C2', 'P21'] #monoclinic

# Structure composition parameters
ELEMENTS = ["C", "N", "O", "Cl"]
#N_ATOMS_LIMS = (10, 30)
#ATOM_VOLUME_START_WIDTH = (14, 8)
N_ATOMS_LIMS = (10, 20)
ATOM_VOLUME_START_WIDTH = (14, 8)

# Resolution limits
d_high = 1.0  # High resolution limit
d_low = 1.5   # Low resolution limit

# Generate number of atoms
n_atoms = sample_gen(range(*N_ATOMS_LIMS))

# Initialize structure generator
str_generator = GenBuilder(
    classname=core.CctbxStr.generate_packing,
    sg=sample_gen(SPACE_GROUPS),
    atoms=sample_gen(ELEMENTS, size=n_atoms),
    atom_volume=distr_gen(sts.uniform(*ATOM_VOLUME_START_WIDTH)),
    seed=utils.distr_gen(sts.randint(1, 2**32-1))
)


def runner(pattern):
    """Process a single crystal structure pattern.
    
    Args:
        pattern: Crystal structure pattern object
        
    Returns:
        dict: Contains Patterson maps, structure parameters and intensity data
    """
    params = pattern.report_params()
    structure = pattern.structure
    
    # Calculate structure factors
    I_high = structure.structure_factors(d_min=d_high).f_calc().sort().as_intensity_array()
    ind_high = np.array(list(I_high.indices()))
    #print(max(list(a_high.indices())))
    I_low = structure.structure_factors(d_min=d_low).f_calc().sort().as_intensity_array()
    ind_low = np.array(list(I_low.indices()))
    #print(max(list(a_low.indices())))
    
    patt_low = pu.calculate_patterson_fft(I_low.data().as_numpy_array(), miller_indices = ind_low, map_shape = (12,12,12))
    patt_high = pu.calculate_patterson_fft(I_high.data().as_numpy_array(), miller_indices = ind_high, map_shape = (24,24,24))
    assert patt_low.min() == 0 and patt_high.min() == 0
    assert patt_low.max() == 1 and patt_high.max() == 1
    #print(params,'\n',I_high.data().size(), patt_high.shape, I_low.data().size(), patt_low.shape,'\n','*'*50)
    
    # Prepare intensity and index data
    I_high = I_high.data().as_numpy_array()
    I_low = I_low.data().as_numpy_array()
    
    return {
        'patt_low': patt_low,
        'patt_high': patt_high,
        'structure_params': params,
        'ind_low': ind_low,
        'ind_high': ind_high
    }


if __name__ == '__main__':
    freeze_support()
    CHANKS = 40
    CHANK_SIZE = 10000
    CORES = 4

    for i in range(CHANKS):
        chank = it.islice(str_generator, CHANK_SIZE)
        pool = mp.Pool(CORES)
        
        with pool as p:
            results = p.map(runner, chank)
            
        #np.savez_compressed(f'clin{d_low}_{d_high}_{i}', db = np.array(results))
        np.savez_compressed(f'patterson_10_15_12/patterson_{i}', db=np.array(results))
        pool.close()