import pandas as pd
import sys

def sum_prediction_column(filename):
    total_sum = 0
    
    for i in range(5, 11):  # Iterating from 5 to 10 (inclusive)
        file_path = f"{filename}_{i}.csv"
        try:
            print(f"Reading {file_path}")
            df = pd.read_csv(file_path, encoding='utf-8')
            if "prediction" in df.columns:
                total_sum += df["prediction"].sum()
                print(f"Sum of 'prediction' column in {file_path}: {df['prediction'].sum()}")
            else:
                print(f"Warning: 'prediction' column not found in {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Total sum of 'prediction' column across all files: {total_sum}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    sum_prediction_column(input_filename)