# Intel Image Classification

## Project Overview

This project classifies natural scene images into six categories:

* Buildings
* Forest
* Glacier
* Mountain
* Sea
* Street

The model was trained using TensorFlow and MobileNetV2 transfer learning on the Intel Image Classification Dataset.

## Files

* Intel_Image_Classification.ipynb – Training and evaluation notebook
* app.py – Streamlit application
* scene_classifier.keras – Trained model
* requirements.txt – Python dependencies

## Run Locally

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

## Dataset

Intel Image Classification Dataset:
https://www.kaggle.com/datasets/puneet6060/intel-image-classification
