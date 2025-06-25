# AIMed - AI-Enhanced Medical Outcome Prediction

A machine learning framework for predicting patient outcomes in Medical Intensive Care Units (MICU) using clinical data, vital signs, and diagnostic codes.

## Overview

AIMed leverages advanced machine learning techniques to predict patient outcomes by analyzing comprehensive clinical features including vital signs, laboratory values, medications, and diagnostic history. The project implements a novel clustering-based approach to identify patient subgroups and train specialized models for improved prediction accuracy.

## Key Features

- **Patient Clustering**: Automatic identification of patient subgroups using K-means and K-modes algorithms
- **Multi-Model Framework**: Implementation of various ML algorithms including:
  - XGBoost
  - Deep Neural Networks
  - Random Forest
  - Logistic Regression
  - Elastic Net
- **Clinical Feature Engineering**: Processing of 60+ clinical features including:
  - Vital signs (heart rate, blood pressure, temperature, respiratory rate)
  - Laboratory values (glucose, albumin, platelets, etc.)
  - Medication dosages (vasopressors, inotropes)
  - Clinical scores (SAPS II, SOFA, LACE)
  - Demographic information
- **ICD-9 Integration**: Incorporation of diagnosis codes with CCS (Clinical Classifications Software) grouping
- **SHAP Analysis**: Model interpretability through SHAP (SHapley Additive exPlanations) values

## Project Structure

```
AIMed/
├── clustering/                 # Patient clustering algorithms and analysis
│   ├── cluster_analysis.py    # Main clustering pipeline
│   ├── kmeans_analysis.py     # K-means implementation
│   └── ICD_to_CCS.py         # ICD-9 to CCS conversion
├── data/                      # Dataset files (not included in repo)
│   ├── MICU_admits_clean.csv # Cleaned MICU admissions data
│   ├── patient_ccs.csv       # Patient CCS codes
│   └── labeled_clustered_data.csv # Clustered patient data
├── models/                    # Model training and evaluation scripts
│   ├── baseline_fit.py       # Baseline model training
│   ├── xg_boost_fit.py       # XGBoost implementation
│   ├── neuralnet_fit.py      # Neural network training
│   ├── all_data_pipeline.py  # Full pipeline execution
│   └── shap_boost.py         # SHAP analysis for XGBoost
├── Hierarchy_Based_Clustering/ # Hierarchical clustering experiments
└── icd9/                      # ICD-9 code processing utilities
```

## Installation

### Prerequisites
- Python 3.7+
- CUDA-capable GPU (recommended for neural network training)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AIMed
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Required Libraries
```
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.23.0
xgboost>=1.2.0
tensorflow>=2.3.0
keras>=2.4.0
matplotlib>=3.3.0
seaborn>=0.11.0
shap>=0.37.0
tqdm>=4.50.0
joblib>=0.17.0
```

## Usage

### 1. Data Preprocessing
Ensure your data is in the correct format with all required clinical features as specified in `data/features.txt`.

### 2. Patient Clustering
```bash
python clustering/cluster_analysis.py --n_clusters 3 --method kmeans
```

### 3. Train Baseline Models
```bash
# Train all baseline models
bash models/train-baselines.sh

# Or train individual models
python models/baseline_fit.py
```

### 4. Train XGBoost Models
```bash
# Train XGBoost on all data
python models/xg_boost_fit_all_data.py

# Train cluster-specific XGBoost models
bash models/train-xgboost_cluster.sh
```

### 5. Train Neural Networks
```bash
# Train neural network models
bash models/train-neuralnetwork.sh

# Or run with specific configurations
python models/neuralnet_fit.py --clusters 3 --epochs 100
```

### 6. Model Interpretation
```bash
# Generate SHAP values for model interpretation
python models/shap_boost.py --model_path xg_boost_cluster_0.dat
```

## Clinical Features

The model uses the following feature categories:

- **Demographics**: Age, gender, marital status, insurance type
- **Vital Signs**: Heart rate, blood pressure, temperature, respiratory rate (min/max/mean)
- **Laboratory Values**: Glucose, albumin, platelets, urea nitrogen, calcium, magnesium
- **Medications**: Dosages of vasopressors (norepinephrine, vasopressin, etc.)
- **Clinical Scores**: SAPS II, SOFA, LACE score
- **Administrative**: Length of stay, number of visits, admission acuity
- **Comorbidities**: Charlson comorbidity index

## Results

Model performance metrics are saved in:
- `models/results/`: Classification reports for each model and cluster
- `models/final_model_results.xlsx`: Comprehensive performance comparison

## Notebooks

Interactive Jupyter notebooks for analysis:
- `models/Pipeline.ipynb`: Complete pipeline walkthrough
- `models/SHAP Analysis.ipynb`: Model interpretation visualizations
- `clustering/Cluster Averages Demographics.ipynb`: Cluster characteristic analysis

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under a restrictive academic license. See [LICENSE](LICENSE) file for details.

## Acknowledgments

- MIMIC-III Database for providing anonymized clinical data
- Clinical Classifications Software (CCS) for diagnosis grouping
- The medical professionals who validated our approach

## Disclaimer

This software is for research purposes only and should not be used for clinical decision-making without proper validation and regulatory approval.