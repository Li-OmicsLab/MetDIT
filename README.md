# MetDIT: Transforming and Analyzing Clinical Metabolomics Data with Convolutional Neural Networks

## Introduction
This paper introduces a new method called MetDIT, designed to effectively analyze intricate metabolomics data using deep convolutional neural networks (CNN). MetDIT comprises two components: TransOmics and NetOmics. Since CNN models have difficulty in processing one-dimensional (1D) sequence data efficiently, we developed TransOmics, a framework that transforms sequence data into two-dimensional (2D) images, while maintaining a one-to-one correspondence between the sequences and images. NetOmics, the second component, leverages a CNN architecture to extract more discriminative repre-sentations from the transformed samples.

![main-architecture](./images/figure-1.png)

## Mian Results
We conduct extensive experiments on three benchmark dataset, including: **CA**, **ISR**, and **Fungal**. The model performance was evaluated using the area under the receiver operating characteristic curve (*AUROC*), *Precision*, *Recall*, and *Accuracy*.

### Results on CA

| Method | Accuarcy | Precision | Recall | F1-Score | 
|--------|----------|-----------|--------|----------| 
| Random Forest | 0.73 | 0.75 | 0.78 | 0.76 |
| SVM | 0.62 | 0.66 | 0.68 | 0.67 | 
| XGBoost | 0.76 | 0.78 | 0.83 | 0.81 | 
| LightGBM | 0.75 | 0.81 | 0.77 | 0.79 |
| MLP | 0.79 | 0.80 | 0.84 | 0.82 | 
| **MetDIT** | **0.93** | **0.93** | **0.93** | **0.93** |


### Results on ISR

| Method | Accuarcy | Precision | Recall | F1-Score | 
|--------|----------|-----------|--------|----------| 
| Random Forest | 0.75 | 0.49 | 0.52 | 0.48 |
| SVM | 0.73 | 0.51 | 0.59 | 0.55 |
| XGBoost | 0.78 | 0.58 | 0.71 | 0.64 |
| LightGBM | 0.76 | 0.54 | 0.70 | 0.61 |
| MLP | 0.79 | 0.65 | 0.78 | 0.71 |
| **MetDIT** | **0.94** | **0.95** | **0.94** | **0.94** |


### Results on Fungal

| Method | Accuarcy | Precision | Recall | F1-Score | 
|--------|----------|-----------|--------|----------| 
| Random Forest | 0.97 | 0.92 | 0.94 | 0.93 | 
| SVM | 0.96 | 0.94 | 0.95 | 0.94 |
| XGBoost | 0.96 | 0.88 | 0.91 | 0.89 | 
| LightGBM | 0.97 | 0.91 | 0.94 | 0.92 |
| MLP | 0.97 | 0.94 | 0.97 | 0.95 | 
| **MetDIT** | **0.98** | **0.96** | **0.96** | **0.96** |


## Environment

- The code is developed using python 3.8 on Ubuntu 20.04.
- The TransOmics and inference NetOmics can run on cpu. 
- If you want to train the NetOmics on custom dataset, Nvidia GPUs are needed. 
- This code is development and tested using one Nvidia A100 GPU with 40GB memory.  


## Quick start

### Installation

1. Clone this repo:
   ```
   git clone https://github.com/Li-OmicsLab/OmicsDIT.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Example of TransOmics

The TransOmics is responsible for transferring metabolom-ic data into 2D images. It contains to parties: data pre-processing and image transfer. 

1. Data pre-processing:
   ```
   python 01-feature_process.py -ofp {your csv data path} -sfp {the path to save} -log -zs
   ```

- You can use the following example to run the demo dataset. 
    ```
    python 01-feature_process.py
    ```

2. Data generation:
    ```
    python 02-convert_by_cols.py -fn {the csv data path} -sp {the path to save converted images} -sz 32 -mt summation
    ```

- You can use the following example to run the demo dataset. 
    ```
    python 02-convert_by_cols.py
    ```
- The converted samples are show as follows (with 128 pix):

   ![cv-1](./images/cv1.png)
   ![cv-2](./images/cv2.png)
   
   ![cv-2](./images/cv3.png)
   ![cv-3](./images/cv4.png)
  -------------

If you want use this code to your own **custom dataset**, please follow the following steps.

1. Create dataset
   
   Our code must be used on the format data to guarantee get the correct results. The data type is shown as follow.

   * The first line is the name of bio-markers.
   * The first row is the gound-truth label, represented by integers (*e.g. 1,2,3,...*).
   * Each line defines a individual sample. 

   ![data-type](./images/seq_data_sample.png)

   

