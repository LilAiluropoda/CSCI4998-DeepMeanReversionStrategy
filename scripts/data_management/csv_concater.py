import pandas as pd
import os

def concat_csv_files(root_dir):
    # Walk through the directory structure
    for root, dirs, files in os.walk(root_dir):
        # Filter CSV files with the specific patterns
        csv_1997_2007 = [f for f in files if f.endswith('19972007.csv')]
        csv_2007_2017 = [f for f in files if f.endswith('20072017.csv')]
        
        # If both files exist in the current directory
        if csv_1997_2007 and csv_2007_2017:
            stock_symbol = os.path.basename(root)
            print(f"Processing {stock_symbol}...")
            
            # Read both CSV files without headers
            # Assuming first column is date
            columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            df1 = pd.read_csv(os.path.join(root, csv_1997_2007[0]), names=columns, header=None)
            df2 = pd.read_csv(os.path.join(root, csv_2007_2017[0]), names=columns, header=None)
            
            # Convert date column to datetime
            df1['Date'] = pd.to_datetime(df1['Date'])
            df2['Date'] = pd.to_datetime(df2['Date'])
            
            # Concatenate the dataframes
            combined_df = pd.concat([df1, df2], axis=0)
            
            # Sort by date
            combined_df = combined_df.sort_values('Date')
            
            # Remove duplicates keeping the first occurrence
            combined_df = combined_df.drop_duplicates(subset=['Date'], keep='first')
            
            # Convert date back to string format YYYY-MM-DD
            combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
            
            # Define output filename
            output_filename = os.path.join(root, f'{stock_symbol}19972017.csv')
            
            # Remove the file if it exists
            if os.path.exists(output_filename):
                try:
                    os.remove(output_filename)
                    print(f"Removed existing file: {output_filename}")
                except Exception as e:
                    print(f"Error removing file: {e}")
            
            # Save the combined dataframe without header
            combined_df.to_csv(output_filename, index=False, header=False)
            print(f"Created: {output_filename}")

# Specify the root directory
root_directory = r"C:\Users\Steve\Desktop\Projects\fyp\app\data\stock_data"

# Run the function
concat_csv_files(root_directory)