# paper_implementations
Implementation of interesting research papers

1) ecg_classification/ecg_classification_model.py

    Implementation of "Cardiologist-Level Arrhythmia Detection with Convolutional Neural Networks", 2017, by
    Pranav Rajpurkar, Awni Y. Hannun, Masoumeh Haghpanahi, Codie Bourn, Andrew Y. Ng
    
    Paper Link: https://arxiv.org/pdf/1707.01836.pdf

    This is a 34-layer deep network to classify single channel ECG data into 14 classes. 12 of these classes represent    
    different   heart arrhythmias in ECG. Remaining two classes represent Noise and normal ECG data.

    Architecture-wise, this is a 1D residual network with 16 residual blocks.

    Code tested with:
    Keras 2.2.0, Tensorflow 1.8.0, Ubuntu 16.04, Python 3.5
  
2) chest_xray_classification/CheXNet.py     (Pneumonia Vs Not-Pneumonia)
   chest_xray_classification/CheXNet14.py   (classify chest-xray into 14 abnormality classes)
   
   Implementation of "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning", 2017, by
   Pranav Rajpurkar, ..., Andrew Y. Ng
   
   Paper Link: https://arxiv.org/pdf/1711.05225.pdf
   
   This is a DenseNet121 (121-layer deep network) to classify chest-xrays into 14 lung abnormalities.
   
   Code tested with:
   Keras 2.2.0, Tensorflow 1.8.0, Ubuntu 16.04, Python 3.5
   
   
   



