# Titanic Data Acquisition, Cleaning, and Preprocessing

## Objective
This repository contains the Python scripts and documentation for acquiring, cleaning, and preprocessing the public Titanic Passenger Survival dataset. The goal is to prepare the raw data for downstream machine learning analysis by handling missing values, inconsistent categorical data, and numerical outliers.

## Files in this Repository
* `main.py`: The main Python script containing the data cleaning and preprocessing pipeline.
* `titanic.csv`: The raw dataset downloaded from Kaggle (Note: download this from Kaggle to run the script).
* `task_report.docx`: Detailed narrative report with screenshots and rationale for preprocessing choices.

## Key Preprocessing Steps
1. **Handling Missing Values:** Dropped the `Cabin` feature (77% missing), imputed `Age` with the median, and imputed `Embarked` with the mode.
2. **Outlier Treatment:** Applied Winsorization (IQR Capping) to the `Fare` column to neutralize extreme outliers without deleting passenger records.
3. **Categorical Encoding:** Applied One-Hot Encoding (with `drop_first=True`) to convert text categories into machine-readable binary columns.
4. **Feature Scaling:** Applied `StandardScaler` to `Age` and `Fare` to optimize the dataset for distance-based ML algorithms.

## How to Run
Ensure you have `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn` installed. Place `titanic.csv` in the root directory and run:
`python main.py`