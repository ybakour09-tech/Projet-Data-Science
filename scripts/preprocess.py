import pandas as pd
import numpy as np

def preprocess_and_clean_all(file_path):
    print(f"Loading raw dataset from: {file_path}")
    df = pd.read_csv(file_path)
    
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")
    
    # 1. Drop the Influencer column
    print("\nDropping 'Influencer' column...")
    df = df.drop(columns=['Influencer'])
    
    # 2. Drop rows with missing values
    print("Dropping rows with missing values (NaN)...")
    df = df.dropna()
    print(f"Shape after dropping NaNs: {df.shape} (removed {initial_shape[0] - df.shape[0]} rows)")
    
    # 3. Detect and remove outliers using IQR for each numeric column
    numeric_cols = ['TV', 'Radio', 'Social Media', 'Sales']
    outlier_mask = pd.Series(False, index=df.index)
    
    print("\nDetecting outliers using IQR method:")
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Identify outliers in this column
        col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = col_outliers.sum()
        print(f"  {col}: Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}, bounds=[{lower_bound:.4f}, {upper_bound:.4f}]")
        print(f"    Outliers found: {outlier_count}")
        
        # Combine with overall mask
        outlier_mask = outlier_mask | col_outliers
        
    total_outliers = outlier_mask.sum()
    print(f"\nTotal rows containing at least one outlier: {total_outliers}")
    
    # Drop rows with outliers
    df_cleaned = df[~outlier_mask]
    print(f"Shape after removing outliers: {df_cleaned.shape} (removed {total_outliers} rows)")
    
    # 4. Save the cleaned dataset
    clean_output_path = "data/Clean_Data_HSS.csv"
    df_cleaned.to_csv(clean_output_path, index=False)
    print(f"\nCleaned and preprocessed dataset saved to: {clean_output_path}")
    print(f"Final dataset shape: {df_cleaned.shape}")

if __name__ == "__main__":
    preprocess_and_clean_all("data/Dummy Data HSS.csv")
