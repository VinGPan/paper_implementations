# paper_implementations
Implementation of interesting research papers

1) ecg_classification/ecg_classification_model.py
Implementation of "Cardiologist-Level Arrhythmia Detection with Convolutional Neural Networks", 2017, by
Pranav Rajpurkar, Awni Y. Hannun, Masoumeh Haghpanahi, Codie Bourn, Andrew Y. Ng
Paper Link: https://arxiv.org/pdf/1707.01836.pdf

This is a 34-layer deep network to classify single channel ECG data into 14 classes. 12 of these classes represent different  heart arrhythmias in ECG. Remaining two classes represent Noise and normal ECG data.

Architecture-wise, this is a 1D residual network with 16 residual blocks.


