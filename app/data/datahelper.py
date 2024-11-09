import pandas as pd
from typing import List
import numpy as np
from pathlib import Path

class DataLoader:
    """
    Handles all data loading, processing, and output operations for financial data.
    
    This class is responsible for:
    - Loading and preprocessing stock data from CSV files (both standard and signal formats)
    - Formatting calculated results
    - Writing processed data to output files
    
    Supports two main CSV formats:
    1. Stock data format: Date, Open, High, Low, Close, Volume, Adj Close
    2. Signal data format: price;signal
    """
    
    def load_stock_data(self, input_path: str) -> pd.DataFrame:
        """
        Reads and processes stock market data from a CSV file into a DataFrame.
        
        Args:
            input_path (str): Path to the input CSV file.
            
        Returns:
            pd.DataFrame: Processed DataFrame containing adjusted stock data with columns:
                         ['date', 'close', 'open', 'high', 'low', 'volume']
                         Returns empty DataFrame if loading fails.
                         
        Note:
            - Automatically reverses data to ensure chronological order
            - Adjusts OHLC prices using adjustment factor
        """
        try:
            columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            
            df = pd.read_csv(
                input_path, 
                header=None, 
                names=columns,
                parse_dates=['Date']
            )
            
            df = df.iloc[::-1].reset_index(drop=True)
            df['adj_factor'] = df['Adj Close'] / df['Close']
        
            df['open'] = df['Open'] * df['adj_factor']
            df['high'] = df['High'] * df['adj_factor']
            df['low'] = df['Low'] * df['adj_factor']
            df['close'] = df['Adj Close']
            df['volume'] = df['Volume']
            df['date'] = df['Date']
            
            return df[['date', 'close', 'open', 'high', 'low', 'volume']].sort_values('date')
            
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")
            return pd.DataFrame()

    def load_signal_data(self, file_path: str) -> pd.DataFrame:
        """
        Reads trading signal data from CSV and returns a pandas DataFrame.
        
        Args:
            file_path (str): Path to the signal CSV file.
            
        Returns:
            pd.DataFrame: DataFrame with 'price' and 'signal' columns.
        """
        try:
            df = pd.read_csv(file_path, delimiter=';', header=None)
            df.columns = ['price', 'signal']
            df['price'] = pd.to_numeric(df['price'])
            df['signal'] = pd.to_numeric(df['signal'])
            return df
        except Exception as e:
            print(f"An error occurred while processing the signal file: {e}")
            return pd.DataFrame()
  
    def format_stock_data(self, df: pd.DataFrame) -> List[str]:
        """
        Formats DataFrame into semicolon-separated strings for output.
        
        Args:
            df (pd.DataFrame): DataFrame containing calculated indicators.
            
        Returns:
            List[str]: List of formatted strings, each containing:
                      - Close price (4 decimal places)
                      - RSI values 1-20 (integer)
                      - SMA50 and SMA200 (2 decimal places)
        """
        output_rows: List[str] = []
        for index, row in df.iterrows():
            output_row: List[str] = [
                f"{row['close']:.4f}",
                *[f"{row[f'rsi{i}']:.0f}" for i in range(1, 21)],
                f"{row['sma50']:.2f}",
                f"{row['sma200']:.2f}"
            ]
            output_rows.append(';'.join(output_row))
        return output_rows
    
    def write_stock_data(self, output_rows: List[str], output_file_path: str) -> None:
        """
        Writes formatted output rows to a file.
        
        Args:
            output_rows (List[str]): List of formatted strings to write.
            output_file_path (str): Path where the output file should be written.
        """
        with open(output_file_path, "w", newline='') as f:
            for row in output_rows:
                f.write(f"{row}\n")

class TestDataGenerator:
    """
    A class to process CSV data and convert it into a format suitable for deep learning model testing.
    
    This processor handles:
    - Reading input CSV files
    - Converting financial data into testing format
    - Generating trend indicators
    - Writing processed data to output files
    """
    
    def __init__(self, input_path: str, output_path: str):
        """
        Initialize the data processor with input and output paths.
        
        Args:
            input_path (str): Path to the input CSV file
            output_path (str): Path where the processed data will be written
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.data: pd.DataFrame = pd.DataFrame()
        
    def process(self) -> None:
        """
        Execute the complete data processing workflow.
        
        This method orchestrates the entire process:
        1. Reading the input file
        2. Converting data to testing format
        3. Writing the processed data
        """
        self._read_input_file()
        testing_data = self._convert_to_testing_format()
        self._write_output_file(testing_data)
        
    def _read_input_file(self) -> None:
        """
        Read and store data from the input CSV file.
        
        The CSV file is expected to have semicolon-separated values with
        technical indicators and SMA values.
        """
        try:
            self.data = pd.read_csv(self.input_path, sep=';', header=None)
        except IOError as e:
            print(f"Error reading input file: {e}")
            self.data = pd.DataFrame()
            
    def _convert_to_testing_format(self) -> List[str]:
        """
        Convert the loaded data into the required testing format.
        
        Returns:
            List[str]: Formatted strings ready for testing data file
        
        Format:
            Each line: "5 1:{indicator_value} 2:{column_index} 3:{trend}"
        """
        def _format_testing_line(indicator_value: str, column_index: int, trend: str) -> str:
            """
            Format a single line of testing data.
            
            Args:
                indicator_value (str): The technical indicator value
                column_index (int): The index of the indicator column
                trend (str): The calculated trend value
                
            Returns:
                str: Formatted line for testing data file
            """
            return f"5 1:{indicator_value} 2:{column_index} 3:{trend}\n"
        formatted_data: List[str] = []
        
        for row_index in range(len(self.data)):
            trend = self._get_trend_from_sma(self.data.iloc[row_index])
            
            # Process each indicator column (excluding price and SMAs)
            for col_index in range(1, len(self.data.columns) - 2):
                formatted_line = _format_testing_line(
                    indicator_value=str(self.data.iloc[row_index, col_index]),
                    column_index=col_index,
                    trend=trend
                )
                formatted_data.append(formatted_line)
                
        return formatted_data
    
    def _get_trend_from_sma(self, row: pd.Series) -> str:
        """
        Calculate trend based on SMA50 and SMA200 values from a data row.
        
        Args:
            row (pd.Series): A row of data containing SMA values
            
        Returns:
            str: "1.0" for uptrend (SMA50 > SMA200), "0.0" for downtrend
        """
        sma50 = float(row[21])
        sma200 = float(row[22])
        return "1.0" if sma50 - sma200 > 0 else "0.0"
    
    def _write_output_file(self, data: List[str]) -> None:
        """
        Write the processed data to the output file.
        
        Args:
            data (List[str]): The formatted testing data to write
        """
        try:
            with open(self.output_path, "w") as writer:
                writer.writelines(data)
        except IOError as e:
            print(f"Error writing output file: {e}")

# Example usage:
if __name__ == "__main__":
    loader = DataLoader()
    
    # For stock market data
    stock_df = loader.load_stock_data("stock_data.csv")
    if not stock_df.empty:
        formatted_data = loader.format_stock_data(stock_df)
        loader.write_stock_data(formatted_data, "output.txt")
    
    # For signal data
    signal_df = loader.load_signal_data("signal_data.csv")
    if not signal_df.empty:
        print(signal_df.head())