import pandas as pd
from typing import List
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class PathConfig:
    """Centralized path configuration"""
    BASE_DIR: Path = Path(os.getenv('BASE_DIR', 'C:/Users/Steve/Desktop/Projects/fyp'))
    DATA_DIR: Path = BASE_DIR / 'app' / 'data' / 'stock_data'
    
    @classmethod
    def get_data_file_path(cls, filename: str) -> Path:
        """Get full path for a data file"""
        return cls.DATA_DIR / filename

# Rest of the code remains the same, just update path references
class DataLoader:
    """
    Handles all data loading, processing, and output operations for financial data.
    """
    
    def load_stock_data(self, input_path: str) -> pd.DataFrame:
        try:
            columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            
            df = pd.read_csv(
                PathConfig.get_data_file_path(input_path), 
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
        try:
            df = pd.read_csv(PathConfig.get_data_file_path(file_path), delimiter=';', header=None)
            df.columns = ['price', 'signal']
            df['price'] = pd.to_numeric(df['price'])
            df['signal'] = pd.to_numeric(df['signal'])
            return df
        except Exception as e:
            print(f"An error occurred while processing the signal file: {e}")
            return pd.DataFrame()
  
    def format_stock_data(self, df: pd.DataFrame) -> List[str]:
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
        with open(PathConfig.get_data_file_path(output_file_path), "w", newline='') as f:
            for row in output_rows:
                f.write(f"{row}\n")

class TestDataGenerator:
    """
    A class to process CSV data and convert it into a format suitable for deep learning model testing.
    """
    
    def __init__(self, input_path: str, output_path: str):
        self.input_path = PathConfig.get_data_file_path(input_path)
        self.output_path = PathConfig.get_data_file_path(output_path)
        self.data: pd.DataFrame = pd.DataFrame()
        
    def process(self) -> None:
        self._read_input_file()
        testing_data = self._convert_to_testing_format()
        self._write_output_file(testing_data)
        
    def _read_input_file(self) -> None:
        try:
            self.data = pd.read_csv(self.input_path, sep=';', header=None)
        except IOError as e:
            print(f"Error reading input file: {e}")
            self.data = pd.DataFrame()
            
    def _convert_to_testing_format(self) -> List[str]:
        def _format_testing_line(indicator_value: str, column_index: int, trend: str) -> str:
            return f"5 1:{indicator_value} 2:{column_index} 3:{trend}\n"
            
        formatted_data: List[str] = []
        
        for row_index in range(len(self.data)):
            trend = self._get_trend_from_sma(self.data.iloc[row_index])
            
            for col_index in range(1, len(self.data.columns) - 2):
                formatted_line = _format_testing_line(
                    indicator_value=str(self.data.iloc[row_index, col_index]),
                    column_index=col_index,
                    trend=trend
                )
                formatted_data.append(formatted_line)
                
        return formatted_data
    
    def _get_trend_from_sma(self, row: pd.Series) -> str:
        sma50 = float(row[21])
        sma200 = float(row[22])
        return "1.0" if sma50 - sma200 > 0 else "0.0"
    
    def _write_output_file(self, data: List[str]) -> None:
        try:
            with open(self.output_path, "w") as writer:
                writer.writelines(data)
        except IOError as e:
            print(f"Error writing output file: {e}")

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