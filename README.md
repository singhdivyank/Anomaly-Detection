# CS6140 ML Project: Analyzing Household Responses to Dynamic Time-of-Use Pricing Signals

## Overview

This project aims to analyze household responses to dynamic Time-of-Use (dToU) pricing signals using machine learning techniques. By leveraging data from London's smart meter energy use dataset, the project identifies abnormal consumption patterns, visualizes key trends, and employs both unsupervised (K-Means clustering) and supervised (Random Forest and Neural Network) learning methods for anomaly detection.

## Folder Structure

```
cs6140_ml_project/
├── README.md                # Project overview and instructions
├── requirements.txt         # List of project dependencies
├── setup.py                 # Setup script for packaging/installing (if needed)
├── data/
│   ├── raw/                 # Original/raw data from the London smart meter dataset
│   └── processed/           # Cleaned and preprocessed data files
├── notebooks/
│   ├── EDA.ipynb            # Exploratory data analysis notebook
│   ├── anomaly_visualization.ipynb  # Notebook for anomaly detection visualization
│   └── model_training.ipynb # Notebook for model training and experiments
├── src/
│   ├── __init__.py          # Marks src as a Python package
│   ├── data_preprocessing.py  # Data cleaning and preprocessing scripts
│   ├── feature_engineering.py # Feature extraction scripts (e.g., time-based features)
│   ├── model_kmeans.py      # K-Means clustering implementation
│   ├── model_random_forest.py  # Random Forest classifier implementation
│   ├── model_neural_network.py # Neural network model implementation
│   ├── evaluation.py        # Model evaluation functions (e.g., silhouette score, accuracy)
│   └── utils.py             # Utility functions (plotting, saving figures, etc.)
├── models/
│   ├── kmeans_model.pkl     # Saved clustering model
│   ├── rf_model.pkl         # Saved Random Forest model
│   └── nn_model.h5          # Saved Neural Network model
├── reports/
│   ├── final_report.pdf     # Final project report with findings and recommendations
│   └── presentation.pdf     # Slides for presenting your project
├── figures/
│   ├── eda_plots/           # Plots and charts from the EDA phase
│   └── clustering_results/  # Visualizations of clustering and anomaly detection
└── docs/
    ├── CS6140_ML_Project_Proposal.pdf  # Project proposal document
    └── meeting_notes.md     # Notes from group meetings and discussions
```

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd cs6140_ml_project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- **Data Preprocessing:**  
  Run `src/data_preprocessing.py` to load, clean, and normalize your raw dataset. Processed data is saved in the `data/processed/` folder.

- **Feature Engineering:**  
  Execute `src/feature_engineering.py` to extract time-based and tariff features from your dataset.

- **Exploratory Data Analysis (EDA):**  
  Open `notebooks/EDA.ipynb` to visualize your dataset, understand key distributions, and perform initial analysis.

- **Model Training and Evaluation:**  
  Use `notebooks/model_training.ipynb` to experiment with K-Means clustering, train the Random Forest classifier, and develop a Neural Network for anomaly detection. Evaluation metrics and visualizations are provided to assess model performance.

- **Visualization of Anomalies:**  
  Open `notebooks/anomaly_visualization.ipynb` to visualize anomalies in energy consumption data over time and inspect clustering results.

## Contact

For further questions or clarifications, please contact the project team:
- Ishan Biswas
- Divyank Singh
- Sri Sai Teja Mettu Srinivas

## License

This project is licensed under the MIT License.
```