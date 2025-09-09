#!/usr/bin/env python3

import pandas as pd
import requests
from io import BytesIO
import re
import os
from pathlib import Path

class DataLoader:
    def __init__(self):
        self.data_dir = Path("./data")
        self.data_dir.mkdir(exist_ok=True)
        
    def download_from_google_drive(self, file_id, destination=None):
        """Download file from Google Drive handling the virus scan confirmation."""
        
        # First, try direct download
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        session = requests.Session()
        response = session.get(url)
        
        # Check if we need to handle virus scan warning
        if "virus scan warning" in response.text.lower():
            # Extract the confirmation token
            token_match = re.search(r'name="confirm" value="([^"]+)"', response.text)
            if token_match:
                confirm_token = token_match.group(1)
                # Use the confirmed download URL
                confirmed_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={confirm_token}"
                response = session.get(confirmed_url)
            else:
                # Try alternative method
                confirmed_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
                response = session.get(confirmed_url)
        
        if response.status_code == 200:
            if destination:
                with open(destination, 'wb') as f:
                    f.write(response.content)
                return destination
            else:
                return response.content
        else:
            raise Exception(f"Failed to download file. Status code: {response.status_code}")
    
    def load_dataset_from_drive(self, file_id="109vhmnSLN3oofjFdyb58l_rUZRa0d6C8"):
        """Load dataset directly from Google Drive."""
        try:
            print("Downloading dataset from Google Drive...")
            content = self.download_from_google_drive(file_id)
            
            # Try to load as parquet first
            try:
                df = pd.read_parquet(BytesIO(content))
                print(f"✓ Successfully loaded parquet: {df.shape[0]} rows, {df.shape[1]} columns")
                return df
            except:
                pass
            
            # Try to load as CSV
            try:
                df = pd.read_csv(BytesIO(content))
                print(f"✓ Successfully loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
                return df
            except:
                pass
            
            # If both fail, save the content and let user know
            temp_file = self.data_dir / "downloaded_dataset"
            with open(temp_file, 'wb') as f:
                f.write(content)
            
            raise Exception(f"Could not parse dataset. Content saved to {temp_file}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def load_dataset_from_path(self, file_path):
        """Load dataset from local file path."""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix.lower() in ['.csv', '.txt']:
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                # Try to auto-detect format
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                if content.startswith(b'PAR1'):
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)
            
            print(f"✓ Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            print(f"Error loading dataset from {file_path}: {e}")
            return None
    
    def get_dataset_info(self, df):
        """Get comprehensive dataset information."""
        if df is None:
            return None
            
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': dict(df.dtypes),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist(),
        }
        
        # Add summary statistics for numeric columns
        if info['numeric_columns']:
            info['numeric_summary'] = df[info['numeric_columns']].describe().to_dict()
        
        return info

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_dataset_from_drive()
    if df is not None:
        info = loader.get_dataset_info(df)
        print("\n=== Dataset Information ===")
        print(f"Shape: {info['shape']}")
        print(f"Columns: {info['columns'][:10]}...")  # Show first 10
        print(f"Data types: {len(info['numeric_columns'])} numeric, {len(info['categorical_columns'])} categorical")
        print(f"Missing values: {sum(info['missing_values'].values())} total")
        print(f"Memory usage: {info['memory_usage'] / 1024 / 1024:.1f} MB") 