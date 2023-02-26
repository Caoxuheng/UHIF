# Universal high spatial resolution hyperspectral imaging using hybrid-resolution image fusion
This is a color space transfer-based super-resolution algorithm for HSI-RGB fusion in coping with unknown spectral degradation. By transferring the HR-RGB to CIE XYZ color space, we employ the CIE color matching function (CMF) as the spectral degradation to reconstruct the HR-HSI, which successfully skip the SRF measurement. To efficiently fuse the HR-XYZ and the LR-HSI, we propose a polynomial fusion model that estimates the ratio matrix between the target HR-HSI and the upsampled LR-HSI. The target HR-HSI is reconstructed by combining the ratio matrix and the unsampled LR-HSI. The quantitative results outperform exisiting SOTA (2014-2021) algorithms.  
***The paper has been submitted to a journal***  

![Introduce]()
# Flowchart
**None**
# Result presentation
Result will be uploaded.
# Guidance
**None**
# Requirements
## Environment
`Python3.8`  
`torch 1.12`,`torchvision 0.13.0`  
`Numpy`,`Scipy`  
*Also, we will create a Paddle version that implements FeafusFormer in AI Studio online for free!*
## Datasets
[`CAVE dataset`](https://www1.cs.columbia.edu/CAVE/databases/multispectral/), 
 [`Preprocessed CAVE dataset`](https://aistudio.baidu.com/aistudio/datasetdetail/147509).
# Note
For any questions, feel free to email me at caoxuhengcn@gmail.com.  
If you find our work useful in your research, please cite our paper ^.^
