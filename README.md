# Lung-Cancer-Image-Classification-with-CNN
Lung Cancer Image Classification with CNN

This project implements a Convolutional Neural Network (CNN) to classify lung cancer images into three categories: Benign, Malignant, and Normal. The dataset consists of grayscale images categorized into these three classes. The CNN model is evaluated using k-fold cross-validation, and various performance metrics are reported.

Project Overview

The primary goal of this project is to build and evaluate a CNN model for classifying lung cancer images. The code includes the following steps:

	1.	Load and preprocess image data.
	2.	Define and train a CNN model.
	3.	Evaluate the model using k-fold cross-validation.
	4.	Generate performance metrics and visualize results.
 
Data Preparation

The dataset used for this project can be downloaded from Mendeley Data - https://data.mendeley.com/datasets/bhmdr45bh2/1

The dataset consists of grayscale images divided into three categories:

	•	Benign
	•	Malignant
	•	Normal

Images are resized to 100x100 pixels and normalized to pixel values between 0 and 1. The dataset is then split into training and testing sets using K-Fold cross-validation.

Model Training

The CNN architecture used is as follows:

	•	Convolutional Layer (32 filters, 3x3 kernel)
	•	Max Pooling Layer (2x2 pool size)
	•	Convolutional Layer (64 filters, 3x3 kernel)
	•	Max Pooling Layer (2x2 pool size)
	•	Convolutional Layer (64 filters, 3x3 kernel)
	•	Flatten Layer
	•	Dense Layer (64 units, ReLU activation)
	•	Dropout Layer (20% dropout rate)
	•	Output Layer (3 units, Softmax activation)

The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function.

Results

After training the model, the following metrics are reported:

	•	Test Accuracy: The accuracy of the model on the test set.
	•	Classification Report: Precision, recall, and F1-score for each class.
	•	Confusion Matrix: Visualization of true vs. predicted labels.
