from typing import List, Dict, Callable, Any
import pandas as pd
import numpy as np
from indicators.rsi import RSIIndicator
from indicators.sma import SMAIndicator

class DataLoader:
    """Handles all data loading, processing, and output operations."""
    
    def load_stock_data(self, input_path: str) -> pd.DataFrame:
        """
        Reads and processes CSV file directly into a DataFrame.
        Handles data reversal and adjustment calculations.
        """
        try:
            # Define column names for the CSV
            columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            
            # Read CSV directly into DataFrame with specified headers
            df = pd.read_csv(
                input_path, 
                header=None, 
                names=columns,
                parse_dates=['Date']
            )
            
            # Reverse the data
            df = df.iloc[::-1].reset_index(drop=True)
            
            # Calculate adjustment factor
            df['adj_factor'] = df['Adj Close'] / df['Close']
            
            # Adjust OHLC values
            df['open'] = df['Open'] * df['adj_factor']
            df['high'] = df['High'] * df['adj_factor']
            df['low'] = df['Low'] * df['adj_factor']
            df['close'] = df['Adj Close']
            df['volume'] = df['Volume']
            df['date'] = df['Date']
            
            # Select and sort final columns
            return df[['date', 'close', 'open', 'high', 'low', 'volume']].sort_values('date')
            
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")
            return pd.DataFrame()
  
    def format_output(self, df: pd.DataFrame) -> List[str]:
        """Formats DataFrame into output strings."""
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
    
    def write_output(self, output_rows: List[str], output_file_path: str) -> None:
        """Writes formatted output to file."""
        with open(output_file_path, "w", newline='') as f:
            for row in output_rows:
                f.write(f"{row}\n")

class TechnicalIndicatorCalculator:
    """Handles calculation of technical indicators."""
    
    def __init__(self):
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
        """Calculates specified technical indicators."""
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
        """Calculates RSI indicators."""
        def _apply_rsi_padding(df: pd.DataFrame, period: int) -> None:
            """Applies custom padding to RSI values."""
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
        """Calculates SMA indicators."""
        for period in params['periods']:
            sma_indicator: SMAIndicator = SMAIndicator(close_prices, period)
            df[f'sma{period}'] = [sma_indicator.calculate(j) for j in range(len(df))]
        return df

class FeatureProcessor:
    """Orchestrates the feature processing workflow."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.indicator_calculator = TechnicalIndicatorCalculator()

    def run_analysis(self, input_file_path: str, output_file_path: str, 
                    features: List[str] = None, 
                    custom_params: Dict[str, Dict[str, Any]] = None) -> None:
        """
        Runs the complete analysis process.
        
        Args:
            input_file_path: Path to the input CSV file
            output_file_path: Path where the output should be written
            features: List of features to calculate
            custom_params: Custom parameters for feature calculation
        """
        # Load and preprocess data directly into DataFrame
        df: pd.DataFrame = self.data_loader.load_stock_data(input_file_path)
        
        if df.empty:
            print("Failed to load data. Aborting analysis.")
            return
        
        # Calculate features
        df = self.indicator_calculator.calculate_features(df, features, custom_params)
        
        # Format and write output
        output_rows: List[str] = self.data_loader.format_output(df)
        self.data_loader.write_output(output_rows, output_file_path)
