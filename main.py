import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def main():
    # 1. Data Acquisition
    print("Loading dataset...")
    df = pd.read_csv("titanic.csv") # Ensure titanic.csv is in the same directory
    
    # 2. Initial Exploration
    print("Initial Data Info:")
    print(df.info())
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # 3. Data Cleaning
    # Drop Cabin due to 77% missing data
    df.drop(columns=['Cabin'], inplace=True)
    
    # Impute Age with median and Embarked with mode
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Drop structurally unhelpful columns
    df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

    # Handle Outliers in Fare using IQR Capping
    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['Fare'] = np.where(df['Fare'] > upper_bound, upper_bound, df['Fare'])
    df['Fare'] = np.where(df['Fare'] < lower_bound, lower_bound, df['Fare'])

    # 4. Data Preprocessing
    # One-Hot Encoding for categorical variables
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Standardize numerical features
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

    print("\nData cleaning and preprocessing complete. Final dataset head:")
    print(df.head())

if __name__ == "__main__":
    main()