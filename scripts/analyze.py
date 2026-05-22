import pandas as pd
import numpy as np
from scipy import stats

def analyze_data(file_path):
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    
    print("\n" + "="*40)
    print("1. BASIC INFORMATION")
    print("="*40)
    print(df.info())
    
    print("\n" + "="*40)
    print("2. MISSING VALUES COUNT")
    print("="*40)
    print(df.isnull().sum())
    
    print("\n" + "="*40)
    print("3. DESCRIPTIVE STATISTICS")
    print("="*40)
    desc = df.describe().T
    desc['median'] = df.median(numeric_only=True)
    desc['skewness'] = df.skew(numeric_only=True)
    desc['kurtosis'] = df.kurtosis(numeric_only=True)
    print(desc[['count', 'mean', 'std', 'min', '50%', 'median', 'max', 'skewness', 'kurtosis']])
    
    print("\n" + "="*40)
    print("4. CORRELATION MATRIX")
    print("="*40)
    print(df.corr(numeric_only=True))
    
    print("\n" + "="*40)
    print("5. OUTLIER DETECTION (IQR METHOD)")
    print("="*40)
    numeric_cols = ['TV', 'Radio', 'Social Media', 'Sales']
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"\nFeature: {col}")
        print(f"  Q1 (25%): {q1:.4f}, Q3 (75%): {q3:.4f}, IQR: {iqr:.4f}")
        print(f"  Acceptable bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
        print(f"  Number of outliers: {len(outliers)}")
        if len(outliers) > 0:
            print(f"  Outliers values range: [{outliers[col].min():.4f} to {outliers[col].max():.4f}]")

if __name__ == "__main__":
    analyze_data("data/Dummy Data HSS.csv")
