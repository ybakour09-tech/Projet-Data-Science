import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visualizations(file_path, output_dir="plots"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    
    # 1. Print influencer balance if column exists
    if 'Influencer' in df.columns:
        print("\n--- Influencer Value Counts (Balance Check) ---")
        inf_counts = df['Influencer'].value_counts()
        inf_pct = df['Influencer'].value_counts(normalize=True) * 100
        for idx in inf_counts.index:
            print(f"{idx}: {inf_counts[idx]} ({inf_pct[idx]:.2f}%)")
    else:
        print("\nNo 'Influencer' column found in this dataset.")
        
    print("\n--- Sales Skewness ---")
    sales_skew = df['Sales'].skew()
    print(f"Sales Skewness: {sales_skew:.4f}")
    if abs(sales_skew) < 0.5:
        print("Sales distribution is approximately symmetric.")
    elif sales_skew >= 0.5:
        print("Sales distribution is moderately or highly right-skewed (skewed positive).")
    else:
        print("Sales distribution is moderately or highly left-skewed (skewed negative).")
        
    # Set the style
    sns.set_theme(style="whitegrid")
    
    # Create a figure with subplots
    has_influencer = 'Influencer' in df.columns
    nrows = 3 if has_influencer else 2
    
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 6 * nrows))
    fig.suptitle(f"Data Distribution Analysis ({os.path.basename(file_path)})", fontsize=18, fontweight='bold', y=0.96)
    
    # Plot histograms for numeric variables
    sns.histplot(df['TV'].dropna(), kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('TV Budget Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('TV Budget')
    
    sns.histplot(df['Radio'].dropna(), kde=True, ax=axes[0, 1], color='lightcoral')
    axes[0, 1].set_title('Radio Budget Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Radio Budget')
    
    sns.histplot(df['Social Media'].dropna(), kde=True, ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('Social Media Budget Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Social Media Budget')
    
    sns.histplot(df['Sales'].dropna(), kde=True, ax=axes[1, 1], color='gold')
    axes[1, 1].set_title('Sales Distribution (Target)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Sales')
    
    # Plot countplot for Influencer if exists
    if has_influencer:
        sns.countplot(x='Influencer', data=df, ax=axes[2, 0], order=['Mega', 'Macro', 'Micro', 'Nano'], palette='viridis')
        axes[2, 0].set_title('Influencer Category Balance Countplot', fontsize=14, fontweight='bold')
        axes[2, 0].set_xlabel('Influencer Type')
        axes[2, 0].set_ylabel('Count')
        
        # Hide the unused 6th subplot
        axes[2, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save output plot
    prefix = "clean_" if not has_influencer else ""
    output_path = os.path.join(output_dir, f"{prefix}data_distribution.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nDistribution plots successfully saved to: {output_path}")
    
    # Generate a separate correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".4f", linewidths=0.5)
    plt.title(f"Correlation Heatmap ({os.path.basename(file_path)})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    corr_path = os.path.join(output_dir, f"{prefix}correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300)
    print(f"Correlation heatmap saved to: {corr_path}")

if __name__ == "__main__":
    # Check if cleaned data exists, default to it, otherwise raw
    target = "data/Clean_Data_HSS.csv"
    if not os.path.exists(target):
        target = "data/Dummy Data HSS.csv"
    generate_visualizations(target)

