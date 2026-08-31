Early Detection of Tropical Deforestation from Satellite NDVI Time Series

Overview

This project investigates the use of satellite-derived NDVI time-series data and machine learning models for tropical deforestation detection. It compares conventional machine learning models, a 1D CNN, and an early-detection approach using only the initial portion of the NDVI time series.

Dataset

The dataset contains 908 spatial samples with:

* Longitude and latitude coordinates
* 24 NDVI observations (ndvi_00 to ndvi_23)
* Three land-cover classes:
    * Forest
    * Old Clearing
    * Deforested

Class distribution:

* Forest: 500
* Old Clearing: 292
* Deforested: 116

Methods

The project includes:

* Exploratory data analysis
* NDVI feature engineering
* Baseline machine learning models
* 1D Convolutional Neural Network (CNN)
* Early deforestation detection using the first 12 NDVI observations
* Spatial analysis of prediction errors

Models

The baseline comparison includes:

* Random Forest
* Extra Trees
* XGBoost
* Multi-Layer Perceptron (MLP)
* Tuned Random Forest
* 1D CNN

Results

The best-performing conventional model was Random Forest, achieving approximately 73.6% accuracy and 68.5% Macro F1.

The CNN achieved approximately 69.2% accuracy and 50.7% Macro F1.

Using only the first 12 NDVI observations, the early-detection model achieved approximately 67.6% accuracy and 54.9% Macro F1.

The early-detection experiment shows that useful discrimination is possible before the complete NDVI time series is available, although identifying the deforested class remains challenging.

Project Structure

Early-Deforestation-NDVI/
│
├── data/
│   ├── ndvi_dataset.csv
│   └── ndvi_features.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_cnn_model.ipynb
│   └── 05_early_detection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── evaluation.py
│
├── results/
│   ├── baseline_model_results.csv
│   ├── cnn_results.csv
│   ├── early_detection_results.csv
│   ├── early_detection_predictions.csv
│   ├── early_detection_error_spatial_map.png
│   ├── model_comparison.csv
│   └── model_comparison_macro_f1.png
│
├── requirements.txt
├── README.md
└── .gitignore

Reproducibility

Install the required Python packages using:

pip install -r requirements.txt

The notebooks in the notebooks/ directory contain the complete workflow from data exploration through model evaluation and early detection analysis.

Key Finding

The experiments indicate that the temporal evolution of NDVI contains useful information for distinguishing forest, old clearing, and deforested areas. However, the relatively low recall for the deforested class highlights the need for improved class balancing, temporal modeling, and spatial-context learning in future work.