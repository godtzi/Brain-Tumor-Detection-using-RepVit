##AI MODEL to Detect Brain Tumor, trained using RepVit##

- AI model to detect brain tumors from brain MRI images. 
- It trained using RepVit that already pretrained.
- The dataset used to make this model are 9069 brain MRI images.
- It is only to detect, not to identify the type of tumor.
- It is also didn't tell the place of the tumor, although we can use the Grad-CAM to identify the location of the tumor. 
- It is achieve 98.01% of accuracy.


**DATASET** 
Data Source:
1. Kaggle by  ARWA BASAL : https://www.kaggle.com/datasets/arwabasal/brain-tumor-mri-detection
2. Mendeley data by Hira et all : https://data.mendeley.com/datasets/zwr4ntf94j/1

**MODEL**

RepVit model are using the "repvit_m1_1" from timm library

the model is exported to some kind of format, like .pt (Pytorch), .onnx (ONNX) in 32 bit and 8 bit version.
I highly recommend using the .pt and 32-bit .onnx version (the original), because the 8-bit onnx has a very low accuracy (around 19% of acurracy)


**MODEL SPECS**
The model achive (the .pt model):
Accuracy  : 0.9801
Precision : 0.9876
Recall    : 0.9876
F1-Score  : 0.9876

Confussion Matrix

True Positive (TP) = 719
True Negative (TN) = 169
False Positive (FP) = 9
False Negatif (FN) = 9

*this metrics are test using testing data that didn't contained in training dataset, the number of images are 906.


// will be updated
*Written and made by Ahmad Ghozi (GITHUB : godtzi)
