# Solving the Phase Problem in Crystallography using AI Methods

![Project Image](./images/tmp_method.png)  

## Overview
This project introduces a novel deep learning framework for solving the phase problem in crystallography. By leveraging advanced neural network architectures, we demonstrate significant improvements in structure factor reconstruction from X-ray diffraction data.

## Background
The X-Ray Diffraction (XRD) phase problem arises from the inability to directly measure phase information during diffraction experiments. When X-rays scatter from a crystal, we can only measure the intensities (modules of structure factors) of the diffracted beams, but not their phases. This missing phase information makes it challenging to directly determine the crystal structure.

In this project, we aim to solve this problem by reconstructing the complete structure factors using deep learning approaches. The key aspects are:

- **Input Data**: We work with measured XRD structure factor modules organized in reciprocal space coordinates (h,k,l)
- **Target**: Complete set of structure factor modules
- **Model Architectures**:
  - UNet: For direct spatial feature extraction and upsampling
  - Transformer: To capture long-range dependencies in reciprocal space
  - GAN: For generating realistic structure factor distributions
- **Training**:
  - Loss Function: Mean Squared Error (MSE)
  - Metrics:
    - MSE for overall reconstruction quality
    - R-factor for crystallographic agreement
    - SSIM for structural similarity

## Methods

### Architecture
The framework employs multiple deep learning approaches:

#### 3D-UNet with Fourier Transform Layers
- Custom architecture combining 3D convolutions with FT operations
- Integrated Fourier Transform and inverse FT layers
- Specialized for processing crystallographic data in reciprocal space

#### Transformer Model
- Processes structure factors in reciprocal space coordinates (h,k,l)
- Architecture parameters:
  - Embedding dimension: 128
  - Depth: 5
  - Number of heads: 4
  - MLP ratio: 4
  - Dropout rates: 0.1

### Evaluation Metrics
Performance is evaluated using three key metrics:
1. Mean Squared Error (MSE) - For overall reconstruction quality
2. R-factor - For crystallographic agreement
3. Structural Similarity Index (SSIM) - For structural similarity assessment

## Results
The framework was tested on a diverse set of monoclinic crystal structures from the Cambridge Structural Database (CSD). Example results demonstrate consistent performance across different structures with:
- MSE values ranging from 0.0007 to 0.005
- R-factors between 0.53 and 0.71
- SSIM scores between 0.43 and 0.69

## Data Processing
The project includes comprehensive data processing pipelines for:
- Structure factor calculation
- Reciprocal space mapping
- Phase reconstruction
- Crystal symmetry handling

## Installation & Usage
1. Clone the repository
2. Install dependencies
3. Configure the test parameters in the config file
4. Run the inference pipeline using the provided notebooks

## License
This project is released under the CC0 1.0 Universal license. This means you can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.

## Citation
If you use this work in your research, please cite:

## Acknowledgments
We thank the 

## Contact
For questions about the implementation or to report issues, please open an issue in the repository.