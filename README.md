<img align="right" src="https://www.spiedigitallibrary.org/images/journals/VolumeCovers/OE_InProgress_270_350.jpg"/>  

# Universal high spatial resolution hyperspectral imaging using hybrid-resolution image fusion  

This is a color space transfer-based super-resolution algorithm for HSI-RGB fusion in coping with unknown spectral degradation. By transferring the HR-RGB to CIE XYZ color space, we employ the CIE color matching function (CMF) as the spectral degradation to reconstruct the HR-HSI, which successfully skip the SRF measurement. To efficiently fuse the HR-XYZ and the LR-HSI, we propose a polynomial fusion model that estimates the ratio matrix between the target HR-HSI and the upsampled LR-HSI. The target HR-HSI is reconstructed by combining the ratio matrix and the unsampled LR-HSI. The quantitative results outperform exisiting SOTA (2014-2021) algorithms. 
  
   
   

# Flowchart
![Flowchart](https://github.com/Caoxuheng/imgs/blob/main/oeuhif.png)  
# Abstract  
By fusing a low spatial resolution hyperspectral image (LR-HSI) and a high spatial resolution RGB image (HR-RGB), hybrid-resolution hyperspectral imaging has been a popular framework for acquiring high spatial resolution hyperspectral image (HR-HSI). Existing fusion methods always employ a known spectral response function (SRF) of the RGB camera to reconstruct the HR-HSI. The SRF is often limited or unavailable in practice, restricting the performance of existing methods. To address this problem, we propose a color space transfer-based fusion strategy that obtains HR-HSI based on a hybrid resolution hyperspectral imaging system without measuring SRF. Specifically, by using clustered-based back propagation neural network (BPNN), the HR-RGB is mapped into the CIE XYZ color space, and the HR-XYZ is obtained. In the CIE XYZ color space, its SRF is known, which successfully skip the SRF measurement. To efficiently fuse the HR-XYZ and the LR-HSI, we propose a polynomial fusion model that estimates the ratio matrix between the target HR-HSI and the upsampled LR-HSI. Finally, the target HR-HSI is reconstructed by combining the ratio matrix and the unsampled LR-HSI. Experimental results on two public data sets and our real-world data sets show that the proposed method outperforms five state-of-the-art fusion methods.  
# Result presentation  
![Simulate](https://github.com/Caoxuheng/imgs/blob/main/uhif_simu.png)
![Real](https://github.com/Caoxuheng/imgs/blob/main/uhif_real.png)  
# Guidance
Add your dataset path in `config.py`  
Run `main.py`  
# Requirements
## Environment
`Python3.8`  
`torch 1.12`,`torchvision 0.13.0`  
`Numpy`,`Scipy`,`opencv-python`,`scikit-learn`  

## Datasets
[`CAVE dataset`](https://www1.cs.columbia.edu/CAVE/databases/multispectral/), 
 [`Preprocessed CAVE dataset`](https://aistudio.baidu.com/aistudio/datasetdetail/147509).
# Note
For any questions, feel free to email me at caoxuhengcn@gmail.com.  
If you find our work useful in your research, please cite our paper ^.^  
**preproduction**  
```python  
@article{article,
author = {Cao, Xuheng and Lian, Yusheng and Liu, Zilong and Zhou, Han and Bin, Wang and Huang, Beiqing and Zhang, Wan},
year = { },
month = { },
pages = { },
title = {Universal high spatial resolution hyperspectral imaging using hybrid-resolution image fusion},
journal = {Optical Engineering},
doi = { }
}

