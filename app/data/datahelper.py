import pandas as pd
from typing import List
import numpy as np
from pathlib import Path
from app.utils.path_config import PathConfig

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
        """Process data and generate test file."""
        self._read_input_file()
        testing_data = self._convert_to_testing_format()
        self._write_output_file(testing_data)
        
    def process_training(self, company: str) -> None:
        """Process data and generate training file."""
        self.input_path = PathConfig.get_data_file_path(f"{company}19972007.csv")
        self._read_input_file()
        training_data = self._convert_to_testing_format()
        self._write_output_file(training_data)
        
    def _read_input_file(self) -> None:
        try:
            print(f"Reading from: {self.input_path}")
            self.data = pd.read_csv(self.input_path, sep=';', header=None)
            print(f"Data shape: {self.data.shape}")
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
        try:
            sma50 = float(row[21])
            sma200 = float(row[22])
            return "1.0" if sma50 - sma200 > 0 else "0.0"
        except Exception as e:
            print(f"Error calculating trend: {e}")
            return "0.0"
    
    def _write_output_file(self, data: List[str]) -> None:
        try:
            print(f"Writing to: {self.output_path}")
            with open(self.output_path, "w") as writer:
                writer.writelines(data)
            print(f"Written {len(data)} lines")
        except IOError as e:
            print(f"Error writing output file: {e}")

class TrainTestSplitter:
    """
    Splits historical stock data into training and testing datasets based on specified time periods.
    Works with raw data without scaling.
    """
    
    def split_data(self, 
                   company_ticker: str,
                   start_year: int,
                   train_years: int,
                   test_years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the raw data into training and testing sets based on specified timeframes.
        
        Args:
            company_ticker (str): Stock ticker symbol
            start_year (int): Year to start the split from
            train_years (int): Number of years for training data
            test_years (int): Number of years for testing data
            
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames
        """
        # Load the raw dataset
        full_data_path = f"{company_ticker}19972017.csv"
        
        try:
            # Read raw data without scaling, specifying data types
            df = pd.read_csv(
                PathConfig.get_data_file_path(full_data_path),
                header=None,
                names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'],
                dtype={
                    'Open': float,
                    'High': float,
                    'Low': float,
                    'Close': float,
                    'Volume': float,
                    'Adj Close': float
                },
                parse_dates=['Date'],
                date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
            )
            
            # Reverse the order if needed (assuming newer dates should be at the bottom)
            df = df.iloc[::-1].reset_index(drop=True)
            
            # Calculate period boundaries
            train_start = pd.Timestamp(f"{start_year}-01-01")
            train_end = pd.Timestamp(f"{start_year + train_years}-12-31")
            test_start = pd.Timestamp(f"{start_year + train_years}-01-01")
            test_end = pd.Timestamp(f"{start_year + train_years + test_years}-12-31")
            
            # Split the data
            train_data = df[(df['Date'] >= train_start) & (df['Date'] <= train_end)]
            test_data = df[(df['Date'] >= test_start) & (df['Date'] <= test_end)]
            
            return train_data, test_data
            
        except Exception as e:
            print(f"Error loading or splitting data: {e}")
            raise
    
    def save_split_data(self,
                       company_ticker: str,
                       start_year: int,
                       train_years: int,
                       test_years: int) -> None:
        """
        Split and save the raw data into separate training and testing CSV files.
        
        Args:
            company_ticker (str): Stock ticker symbol
            start_year (int): Year to start the split from
            train_years (int): Number of years for training data
            test_years (int): Number of years for testing data
        """
        try:
            train_data, test_data = self.split_data(
                company_ticker,
                start_year,
                train_years,
                test_years
            )
            
            # Save the split datasets
            train_file = f"{company_ticker}_train.csv"
            test_file = f"{company_ticker}_test.csv"
            
            # Save without index, keeping the same format as input
            train_data.to_csv(
                PathConfig.get_data_file_path(train_file), 
                index=False, 
                header=False,
                date_format='%Y-%m-%d'
            )
            test_data.to_csv(
                PathConfig.get_data_file_path(test_file), 
                index=False, 
                header=False,
                date_format='%Y-%m-%d'
            )
            
            print(f"Successfully split data for {company_ticker}:")
            print(f"Training data ({train_data.shape[0]} rows): {train_file}")
            print(f"Testing data ({test_data.shape[0]} rows): {test_file}")
            
        except Exception as e:
            print(f"Error splitting data for {company_ticker}: {e}")
            
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