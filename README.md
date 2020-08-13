# BraTS_with_SMP
Tumor segmentation using pytorch and pipeline tools from the smp library. 

This code was written as an educational project for the course Deep Learning for Medical Images. 
The data was contributed by the BraTS 2019 challenge: http://braintumorsegmentation.org/ and the pipeline was based on the SMP library https://github.com/qubvel/segmentation_models.pytorch. The ideas, inspiration and comparison were all with the help of the reference list at the bottom.


The BraTS dataset comes in an archive of folders with nii.gz files. This pipeline assumes that the data is extracted, in folders for each scan and divided to image slices. The paths for the images are in the pickle files.
In order to get most of the MRI image, 3 of the different MRI modes (T1c, T2 and flair) were concatenated in 3 channels to make the image fit in the dimensions of an RGB image.
An example to the resulting image, the ground truth mask and the predicted segmented image:
![example image](/example_image_59.PNG)

The Project_Main_Pipeline notebook should be executed in order with a few caveats:
- In the "Choose Network Parameters" cell, if you don't have the first pre-trained model saved in the drive, you should change the "false" to "true". after that, the model can be loaded instead of downloaded from the start.
- The "Train the Model" cell has a few choices: 
    - Number of epochs
    - Batch size
    - The learning rate is defined in the cell above, together with the optimizer type and activation
    - Whether or not to start training from an existing checkpoint (if so, changing the addCounter to the current checkpoint number is recommended)
    - Whether or not to use cross-validation (if NOT, then the batch size is defined in the cell above, together with creating train/val dataloaders)

Each cell has documentation to explain it's purpose.

Our trained models can be found at: https://drive.google.com/drive/folders/19WIXhbhj_KB-Ub6vcvSrpxjLajZhhsIF?usp=sharing

The 3-channel dataset can be found at:
 
HGG: https://drive.google.com/drive/folders/1OpVgtILay0e-B0hxSIc3oJwIQFXw-vB3?usp=sharing

LGG: https://drive.google.com/drive/folders/1LQst8g-Au5Cqxf8C0uKQrR85eZIPzaeu?usp=sharing

For any questions we're available at amitgalor@mail.tau.ac.il and doron1336@gmail.com


References:
Chaurasia, A. (2017). LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation. arxiv.org/abs/1707.03718v1.
Chen, L.-C. (2017). Rethinking Atrous Convolution for Semantic Image Segmentation. arxiv.org/abs/1706.05587v3.
Dong, H. (2017). Automatic Brain Tumor Detection and Segmentation Using U-Net Based Fully Convolutional Networks. arxiv.org/abs/1705.03820v3.
Havaei, M. (2016). Brain Tumor Segmentation with Deep Neural Networks. arxiv.org/abs/1505.03540v3.
Isensee, F. (2019). No New-Net. MICCAI - BrainLes 2018. Springer, Cham.
Myronenko, A. (2018). 3D MRI brain tumor segmentation using autoencoder regularization. arxiv.org/abs/1810.11654v3.
Ronneberger, O. (2015). U-Net: Convolutional Networks for Biomedical. arxiv.org/abs/1505.04597v1.
Srinivas, B. (2020). Segmentation of Multi-Modal MRI Brain Tumor Sub-Regions Using Deep Learning. Journal of Electrical Engineering & Technology, 1899–1909.
Xie, S. (2017). Aggregated Residual Transformations for Deep Neural Networks. arxiv.org/abs/1611.05431v2.
Zhao, H. (2017). Pyramid Scene Parsing Network. CVPR, arxiv.org/abs/1612.01105v2.
