#!/usr/bin/env python3

from data_loader import DataLoader
import pandas as pd

def explore_dataset():
    loader = DataLoader()
    df = loader.load_dataset_from_drive()
    
    if df is None:
        print("Failed to load dataset")
        return
    
    print('=== FULL DATASET EXPLORATION ===')
    print(f'Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print()
    
    print('=== SAMPLE DATA ===')
    print(df.head())
    print()
    
    print('=== DATA TYPES ===')
    for col, dtype in df.dtypes.items():
        print(f'{col}: {dtype}')
    print()
    
    print('=== MISSING VALUES ===')
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    for col in df.columns:
        print(f'{col}: {missing[col]:,} ({missing_pct[col]}%)')
    print()
    
    print('=== UNIQUE VALUES ===')
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f'{col}: {unique_count:,} unique values')
        if unique_count < 20 and unique_count > 0:
            sample_values = sorted(df[col].dropna().unique())[:10]
            print(f'  Values: {sample_values}')
        elif unique_count > 0:
            sample_values = sorted(df[col].dropna().unique())[:5]
            print(f'  Sample: {sample_values}...')
        print()
    
    # Save a sample for quick testing
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    sample_df.to_csv('./data/sample_data.csv', index=False)
    print(f"✓ Saved sample data (1000 rows) to ./data/sample_data.csv")
    
    return df

if __name__ == "__main__":
    df = explore_dataset() 