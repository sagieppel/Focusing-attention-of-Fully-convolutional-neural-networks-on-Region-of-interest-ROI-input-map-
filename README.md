# Focusing attention of Fully convolutional neural networks on Region of interest (ROI) input map, using the valve filters method. 

## This is an old code. Newer better version can be found here: [https://github.com/sagieppel/Segmenting-givne-region-of-an-image-using-neural-net-with-ROI-attention-input/](https://github.com/sagieppel/Segmenting-givne-region-of-an-image-using-neural-net-with-ROI-attention-input/blob/main/README.md)

This project contains code for a fully convolutional neural network (FCN) for semantic segmentation with a region of interest (ROI) map as an additional input (figure 1). The net receives image and ROI as a binary map with pixels corresponding to ROI marked 1, and produce pixel-wise annotation of the ROI region of the image.  This code was tested on for semantic segmentation task of materials in transparent vessels where the vessel area of the image was set as the ROI. 
The method is discussed in the paper: [Setting an attention region for convolutional neural networks using region selective features, for recognition of materials within glass vessels](https://arxiv.org/abs/1708.08711)
![](/Figure1.jpg)
Figure 1) Convolutional neural nets (Convnet) with ROI map as input

## General approach for using ROI input in  CNN (valve filter method)
Convolutional neural networks have emerged as the leading methods in detection classification and segmentation of images. Many problems in image recognition require the recognition to be performed only on a specific predetermined region of interest (ROI) in the image. One example of such a case is the recognition of the contents of glass vessels such as bottles or jars, where the glassware region in the image is known and given as the ROI input (Figure 1). Directing the attention of a convolutional neural net (CNN) to a given ROI region without loss of background information is a major challenge in this case. This project uses a valve filter approach to focus the attention of a fully convolutional neural net (FCN) on a given ROI in the image. The ROI mask is inserted into the CNN, along with the image in the form of a binary map, with pixels belonging to the ROI set to one and the background set to zero. The processing of the ROI in the net is done using the valve filter approach presented in Figure 2. In general, for each filter that acts on the image, a corresponding valve filter exists that acts on (convolves) the ROI map (Figure 2). The output of the valve filter convolution is multiplied element-wise with the output of the image filter convolution, to give a normalized feature map (Figure 2). This map is used as input for the next layers of the net. In this case, the net is a standard fully convolutional net (FCN) for semantic segmentation (pixel-wise classification). Valve filters can be seen as a kind of valve that regularizes the activation of image filters in different regions of the image. 


![](/Figure2.png)
Figure 2) The valve filter approach for introduction of ROI map as input to ConvNets. The image and the ROI input are each passed through a separate convolution layer to give feature map and Relevance map, respectively. Each element in the features map is multiplied by the corresponding element in the feature map to give a normalized features map that passed (after RELU) as input for the next layer of the net.

## Requirements
This network was run and trained with Python 3.6  Anaconda package and Tensorflow 1.1. The training was done using Nvidia GTX 1080, on Linux Ubuntu 16.04.

## Setup
1) Download the code from the repository.
2) Download a pre-trained vgg16 net and put in the /Model_Zoo subfolder in the main code folder. A pre-trained vgg16 net can be download from here[https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing] or from here [ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy]

## Tutorial

### Training network:
Run: Train.py

### Prediction using trained network
### (pixelwise classification and segmentation of images)
Run: Inference.py

### Evaluating net performance using intersection over union (IOU):
Run: Evaluate_Net_IOU.py

### Notes and issues
See the top of each script for an explanation as for how to use it.

## Detail valve filters implementation.
The detail  implementation of the valve filters  given in Figures 2 and described below:

1) The ROI map is inserted to the net along with the image. The ROI map is represented as a binary image with pixels corresponding to ROI marked 1 and the rest marked 0. 
2) A set of image filters is convolved (with bias addition) with the image to give a feature map. 
3) A set of valve filters convolved with the ROI map to give a relevance map with the same size and dimension as the feature map (again with bias addition).
4) The feature map is multiplied element wise by the relevance map. Hence,  Each element in the relevance map is multiplied by the corresponding element in the feature map to give normalized feature map. 
5) The normalized feature map is then passed through a Rectified Linear Unit (ReLU)  which zero out any negative map element. The output is used as input for the next layer of the net.  

The net, in this case, is standard fully convolutional neural net for semantic segmentation.
In this way each valve filter act as kind of a valve that regulates the activation the corresponding image filter in different regions of the image. Hence, the valve filter will inhibit some filters in the background zone and others in the ROI zone. 
The valve filters weights are learned by the net in the same way the image filters are learned. Therefore the net learns both the features and the region for which they are relevant.   
In the current implementation, the valve filter act only on the first layer of the convolutional neural net and the rest of the net remained unchanged. 

## Details input/output
The input for the net (Figure 1) are RGB image and ROI map the ROI map is a 2d binary image with pixels corresponding to ROI marked 1 and background marked 0.
The net produce pixel wise annotation as a matrix in size of the image with the value of each pixel is the pixel label (This should be the input in training).

## Background information
The net is based on fully convolutional neural net described in the paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf).  The code is based on 
https://github.com/shekkizh/FCN.tensorflow by Sarath Shekkizhar with encoder  replaced to VGG16. The net is based on the pre-trained VGG16 model by Marvin Teichmann

## Newer models for segmenring materials in vessels
For newer much stronger models for detecting/segmenting matarials in vessels see:
https://github.com/sagieppel/Detecting-and-segmenting-and-classifying-materials-inside-vessels-in-images-using-convolutional-net
 

## Supporting datasets
The net was tested on a [dataset of annotated images of materials in glass vessels](https://github.com/sagieppel/Materials-in-Vessels-data-set). The glass vessel region in the image was taken as the ROI map.
This dataset can be downloaded from  https://drive.google.com/file/d/0B6njwynsu2hXRFpmY1pOV1A4SFE/view?usp=sharing
