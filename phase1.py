from typing import List, Dict, Callable, Any
import pandas as pd
import numpy as np
from indicators.rsi import RSIIndicator
from indicators.sma import SMAIndicator

class DataLoader:
    """
    Handles all data loading, processing, and output operations for stock market data.
    
    This class is responsible for:
    - Loading and preprocessing stock data from CSV files
    - Formatting calculated results
    - Writing processed data to output files
    
    The expected CSV format is:
    Date, Open, High, Low, Close, Volume, Adj Close
    """
    
    def load_stock_data(self, input_path: str) -> pd.DataFrame:
        """
        Reads and processes stock data from a CSV file into a DataFrame.
        
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
            print(f"counterRow: {index + 1}")
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

class TechnicalIndicatorCalculator:
    """
    Handles calculation of technical indicators for stock market data.
    
    Supports multiple technical indicators with customizable parameters.
    Currently implemented indicators:
    - RSI (Relative Strength Index)
    - SMA (Simple Moving Average)
    """
    
    def __init__(self):
        """
        Initializes the calculator with default feature calculators and parameters.
        """
        self.feature_calculators: Dict[str, Callable] = {
            'rsi': self._calculate_rsi_features,
            'sma': self._calculate_sma_features
        }
        
        self.default_params: Dict[str, Dict[str, Any]] = {
            'rsi': {'periods': range(1, 21)},
            'sma': {'periods': [50, 200]}
        }
    
    def calculate_features(self, df: pd.DataFrame, 
                         features: List[str] = None, 
                         custom_params: Dict[str, Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Calculates specified technical indicators for the given data.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing price data.
            features (List[str], optional): List of features to calculate. 
                                          Defaults to all available features.
            custom_params (Dict[str, Dict[str, Any]], optional): Custom parameters for calculations.
                                                                Overrides default parameters.
        
        Returns:
            pd.DataFrame: DataFrame with additional columns for calculated indicators.
        """
        close_prices: np.ndarray = df['close'].values
        
        features = features or list(self.feature_calculators.keys())
        params = {**self.default_params}
        if custom_params:
            params.update(custom_params)
        
        for feature in features:
            if feature in self.feature_calculators:
                df = self.feature_calculators[feature](df, close_prices, params[feature])
        
        return df
    
    def _calculate_rsi_features(self, df: pd.DataFrame, close_prices: np.ndarray, 
                              params: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculates RSI indicators for specified periods with padding.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            close_prices (np.ndarray): Array of closing prices.
            params (Dict[str, Any]): Parameters for RSI calculation.
        
        Returns:
            pd.DataFrame: DataFrame with added RSI columns.
        """
        def _apply_rsi_padding(df: pd.DataFrame, period: int) -> None:
            """
            Applies custom padding to RSI values.
            
            Args:
                df (pd.DataFrame): DataFrame to pad.
                period (int): RSI period being padded.
            """
            df.loc[0, f'rsi{period}'] = 0.0
            df.loc[1, f'rsi{period}'] = 100.0
            
            for j in range(2, len(df)):
                if j < period:
                    df.loc[j, f'rsi{period}'] = df.loc[j, f'rsi{j}']
                    
        for period in params['periods']:
            rsi_indicator: RSIIndicator = RSIIndicator(close_prices, period)
            df[f'rsi{period}'] = [rsi_indicator.calculate(j) for j in range(len(df))]
            _apply_rsi_padding(df, period)
        return df
    
    def _calculate_sma_features(self, df: pd.DataFrame, close_prices: np.ndarray, 
                              params: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculates SMA indicators for specified periods.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            close_prices (np.ndarray): Array of closing prices.
            params (Dict[str, Any]): Parameters for SMA calculation.
        
        Returns:
            pd.DataFrame: DataFrame with added SMA columns.
        """
        for period in params['periods']:
            sma_indicator: SMAIndicator = SMAIndicator(close_prices, period)
            df[f'sma{period}'] = [sma_indicator.calculate(j) for j in range(len(df))]
        return df

class FeatureMaker:
    """
    Orchestrates the complete feature processing workflow for stock market data.
    
    This class coordinates:
    - Data loading and preprocessing
    - Technical indicator calculations
    - Output formatting and writing
    """
    
    def __init__(self):
        """
        Initializes the processor with required components.
        """
        self.data_loader = DataLoader()
        self.indicator_calculator = TechnicalIndicatorCalculator()

    def run_analysis(self, input_file_path: str, output_file_path: str, 
                    features: List[str] = None, 
                    custom_params: Dict[str, Dict[str, Any]] = None) -> None:
        """
        Runs the complete analysis process from data loading to output generation.
        
        Args:
            input_file_path (str): Path to the input CSV file.
            output_file_path (str): Path where the output should be written.
            features (List[str], optional): List of features to calculate.
            custom_params (Dict[str, Dict[str, Any]], optional): Custom parameters for calculations.
        
        Note:
            If data loading fails, the analysis will be aborted.
        """
        df: pd.DataFrame = self.data_loader.load_stock_data(input_file_path)
        
        if df.empty:
            print("Failed to load data. Aborting analysis.")
            return
        
        df = self.indicator_calculator.calculate_features(df, features, custom_params)
        output_rows: List[str] = self.data_loader.format_stock_data(df)
        self.data_loader.write_stock_data(output_rows, output_file_path)
