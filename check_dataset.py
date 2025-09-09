#!/usr/bin/env python3

import pandas as pd
import requests
from io import BytesIO
import sys

def check_dataset():
    print('Checking dataset from Google Drive...')
    url = 'https://drive.google.com/uc?export=download&id=109vhmnSLN3oofjFdyb58l_rUZRa0d6C8'
    
    try:
        response = requests.get(url)
        print(f'Status: {response.status_code}')
        print(f'Content-Type: {response.headers.get("content-type", "unknown")}')
        print(f'Content-Length: {len(response.content)} bytes')
        
        # Try to detect file type
        if response.content.startswith(b'PAR1'):
            print('Detected: Parquet file')
            try:
                df = pd.read_parquet(BytesIO(response.content))
                print(f'Successfully loaded parquet: {df.shape[0]} rows, {df.shape[1]} columns')
                print(f'Columns: {list(df.columns)[:10]}')  # Show first 10 columns
                print(f'Data types:')
                for col, dtype in df.dtypes.items():
                    print(f'  {col}: {dtype}')
                return df
            except Exception as e:
                print(f'Error reading parquet: {e}')
                
        elif b',' in response.content[:1000] or response.content.startswith(b'"'):
            print('Detected: Likely CSV file')
            try:
                df = pd.read_csv(BytesIO(response.content))
                print(f'Successfully loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns')
                print(f'Columns: {list(df.columns)[:10]}')  # Show first 10 columns
                print(f'Sample data:')
                print(df.head(3))
                print(f'Data types:')
                for col, dtype in df.dtypes.items():
                    print(f'  {col}: {dtype}')
                return df
            except Exception as e:
                print(f'Error reading CSV: {e}')
        else:
            print('Unknown file format')
            print(f'First 200 bytes: {response.content[:200]}')
            
    except Exception as e:
        print(f'Error: {e}')
    
    return None

if __name__ == "__main__":
    df = check_dataset() 