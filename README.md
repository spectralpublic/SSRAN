# SSRAN

# Spectral Super-Resolution of Multispectral Images Using Spatial–Spectral Residual Attention Network


### 1. Introduction

This is the reserch code of the IEEE Transactions on Geoscience and Remote Sensing 2022 paper.

[X. Zheng, W. Chen and X. Lu, "Spectral Super-Resolution of Multispectral Images Using Spatial–Spectral Residual Attention Network," IEEE Transactions on Geoscience and Remote Sensing, vol. 60, 2022.](https://ieeexplore.ieee.org/document/9519844)

The spectral super-resolution of multispectral image (MSI) refers to improving the spectral resolution of the MSI to obtain the hyperspectral image (HSI). Most recent works are based on the sparse representation to unfold the MSI into the 2-D matrix in advance for subsequent operations, which results in that the spatial information of MSI cannot be fully explored. In this article, a spatial–spectral residual attention network (SSRAN) is proposed to simultaneously explore the spatial and spectral information of MSI for reconstructing the HSI. The proposed SSRAN is composed of the feature extraction part, the nonlinear mapping part, and the reconstruction part. Firstly, the multispectral features of the input MSI are extracted in the feature extraction part. Second, in the nonlinear mapping part, the spatial–spectral residual blocks are proposed to explore spatial and spectral information of MSI for mapping the multispectral features to the hyperspectral features. Finally, in the reconstruction part, a 2-D convolution is used to reconstruct the HSI from the hyperspectral features. Also, a neighboring spectral attention module is specially designed to explicitly constrain the reconstructed HSI to maintain the correlation among neighboring spectral bands. The proposed SSRAN outperforms the state-of-the-art methods on both simulated and real databases.

### 2. Start


Requirements:
             
	Python
	
	tensorflow
	
	scipy
	
	pandas
	
	numpy

Run "python SSRAN.py" for training and testing.

### 3. Related work

If you find the code and dataset useful in your research, please consider citing our paper.


[X. Zheng, W. Chen and X. Lu, "Spectral Super-Resolution of Multispectral Images Using Spatial–Spectral Residual Attention Network," IEEE Transactions on Geoscience and Remote Sensing, vol. 60, 2022.](https://ieeexplore.ieee.org/document/9519844)

