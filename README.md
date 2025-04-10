# CS6140 ML Project: Analyzing Household Energy Consumption Anomalies

## Overview

This project analyzes London's smart meter energy usage data to identify abnormal consumption patterns. It employs unsupervised anomaly detection using **Isolation Forest** and supervised learning techniques (**LightGBM** and **Neural Networks**) to classify these anomalies based on engineered features. The goal is to understand household responses and detect deviations from typical energy usage.

## Objectives and Expected Outcomes

### Objectives
- Visualize household energy consumption over time and identify anomalies using data visualization techniques.
- Apply unsupervised learning (Isolation Forest) to detect abnormal responses to dynamic Time-of-Use (dToU) pricing.
- Train supervised models (LightGBM, Neural Networks) to classify anomalies based on engineered features.
- Evaluate model performance and compare with visualization insights.
- Provide actionable insights for policymakers to optimize dToU pricing strategies considering inefficiency, non-adherence, or behavioral factors.

### Expected Outcomes
- Identification of distinct clusters or patterns of household behavior.
- Quantification of the proportion of households exhibiting abnormal responses to dToU pricing.
- Visualizations such as time-series plots and anomaly distributions.
- Summarized findings, model performance metrics, and recommendations for future work or policy adjustments.

## Folder Structure

```
Anomaly-Detection/
├── README.md                # Project overview and instructions
├── requirements.txt         # List of project dependencies
├── data/
│   ├── raw/                 # Original/raw data (e.g., smart_meter_data.csv)
│   └── processed/           # Filtered, cleaned, and feature-engineered data files
├── notebooks/
│   ├── EDA.ipynb            # Exploratory data analysis notebook
│   ├── model_training.ipynb # Notebook for loading models and demonstrating predictions
│   └── anomaly_visualization.ipynb  # Notebook for visualizing anomalies and model results
├── src/
│   ├── __init__.py          # Marks src as a Python package
│   ├── filteration.py       # Script to filter the raw dataset (if needed)
│   ├── data_preprocessing.py  # Data cleaning and preprocessing script
│   ├── feature_engineering.py # Feature extraction script (e.g., time-based features)
│   ├── model_isolation_forest.py # Isolation Forest for unsupervised anomaly detection
│   ├── model_lightgbm.py      # LightGBM classifier implementation
│   ├── model_neural_network.py # Neural network model implementation
│   ├── evaluation.py        # Model evaluation functions
│   └── utils.py             # Utility functions
├── models/                  # Saved trained models
│   ├── isolation_forest_model.pkl
│   ├── lightgbm_model.pkl
│   ├── nn_model.h5
│   └── nn_scaler.pkl
└── docs/                    # Project documentation (proposal, etc.)
    └── CS6140_ML_Project_Proposal.pdf
```
*(Note: Other files like `model_kmeans.py`, `model_random_forest.py`, `.py` versions of notebooks, etc., might exist but are secondary to the main workflow described here)*

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Anomaly-Detection
    ```

2.  **Environment Setup:** This project requires a Python environment with the necessary packages installed. You can use conda, venv, or any other environment manager. **Make sure to activate your environment before running the scripts.**

3.  **Install Dependencies:** Install the required Python packages into your environment.
    ```bash
    python -m pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` includes `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `lightgbm`, `matplotlib`, `seaborn`, `joblib`)*

## Workflow

Execute the following steps in order using your activated Python environment.

1.  **Download Data:**
    *   Download the full dataset from [London Datastore](https://data.london.gov.uk/download/smartmeter-energy-use-data-in-london-households/3527bf39-d93e-4071-8451-df2ade1ea4f2/LCL-FullData.zip).
    *   Unzip the contained CSV file (`block_*.csv`) into the `data/raw/` directory.
    *   **Rename** the unzipped CSV file to `smart_meter_data.csv`.

2.  **Filter Raw Data (Optional but Recommended):**
    *   The full dataset is very large. Run the filtering script to create a smaller, manageable subset (e.g., ~1 million rows).
        ```bash
        python src/filteration.py
        ```
    *   **Note:** This script should be configured to read `data/raw/smart_meter_data.csv` and save the filtered output (e.g., potentially overwriting the original or saving to `data/raw/smart_meter_data_filtered.csv`). Adjust subsequent steps accordingly if the filename changes.
    *   **Important:** Ensure the subsequent scripts (`data_preprocessing.py`, etc.) are configured to read the **correct** (potentially filtered) raw data file.

3.  **Preprocess Data:**
    *   Clean, handle missing values, and normalize the (filtered) raw data.
        ```bash
        python src/data_preprocessing.py
        ```
    *   *(Input: `data/raw/smart_meter_data.csv` (or filtered file), Output: `data/processed/smart_meter_data_processed.csv`)*

4.  **Engineer Features:**
    *   Extract time-based features (hour, day of week, etc.).
        ```bash
        python src/feature_engineering.py
        ```
    *   *(Input: `data/processed/smart_meter_data_processed.csv`, Output: `data/processed/smart_meter_data_features.csv`)*

5.  **Unsupervised Anomaly Detection:**
    *   Run Isolation Forest to generate anomaly labels.
        ```bash
        python src/model_isolation_forest.py
        ```
    *   *(Input: `data/processed/smart_meter_data_features.csv`, Output: `data/processed/smart_meter_data_anomalies_if.csv` (with 'anomaly' column), Model: `models/isolation_forest_model.pkl`)*

6.  **Supervised Model Training:**
    *   Train models to predict the anomalies identified by Isolation Forest. Run either or both:
    *   **LightGBM:**
        ```bash
        python src/model_lightgbm.py
        ```
        *(Input: `data/processed/smart_meter_data_anomalies_if.csv`, Model: `models/lightgbm_model.pkl`)*
    *   **Neural Network:**
        ```bash
        python src/model_neural_network.py
        ```
        *(Input: `data/processed/smart_meter_data_anomalies_if.csv`, Model: `models/nn_model.h5`, Scaler: `models/nn_scaler.pkl`)*

7.  **Analysis and Visualization (Notebooks):**
    *   Use Jupyter notebooks to explore the data and model results. Ensure the notebook kernel uses your project environment's Python interpreter.
    *   **`notebooks/EDA.ipynb`**: Explore the processed data and anomaly distributions.
    *   **`notebooks/model_training.ipynb`**: Load the trained models (`IsolationForest`, `LGBMClassifier`, `TensorFlow/Keras`) and demonstrate making predictions.
    *   **`notebooks/anomaly_visualization.ipynb`**: Visualize the detected anomalies, anomaly scores, and compare model predictions.
