from abc import ABC, abstractmethod
from typing import List, Dict, Callable, Any
import csv
import pandas as pd
import numpy as np
from indicators.rsi import RSIIndicator
from indicators.sma import SMAIndicator
from dataclasses import dataclass

@dataclass
class SharePrice:
    """Represents a single day's stock price data."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: float

class DataLoader:
    """Handles data loading and initial preprocessing."""
    
    def read_and_reverse_csv(self, input_path: str) -> List[SharePrice]:
        """Reads and reverses stock price data from a CSV file."""
        share_prices: List[SharePrice] = []
        
        try:
            with open(input_path, 'r') as input_file:
                lines = input_file.readlines()
                
            for line in reversed(lines):
                try:
                    share_price = self._parse_line(line)
                    share_prices.append(share_price)
                except (ValueError, IndexError) as e:
                    print(f"Skipping row due to error: {e}")
                    
        except IOError as e:
            print(f"An error occurred while processing the file: {e}")
            
        return share_prices
    
    def _parse_line(self, line: str) -> SharePrice:
        """Parses a CSV line into a SharePrice object."""
        cleaned_line = line.strip().replace('"', '')
        row = cleaned_line.split(',')
        
        date: str = row[0]
        adj_close: float = float(row[6])
        adj_factor: float = adj_close / float(row[4])
        
        return SharePrice(
            date=date,
            open=float(row[1]) * adj_factor,
            high=float(row[2]) * adj_factor,
            low=float(row[3]) * adj_factor,
            close=adj_close,
            volume=float(row[5]),
            adj_close=adj_close
        )

class DataFrameBuilder:
    """Handles creation and manipulation of DataFrames."""
    
    def create_base_dataframe(self, share_prices: List[SharePrice]) -> pd.DataFrame:
        """Creates and initializes the base DataFrame from share price data."""
        df: pd.DataFrame = pd.DataFrame(
            [(sp.date, sp.close, sp.open, sp.high, sp.low, sp.volume) 
             for sp in share_prices],
            columns=['date', 'close', 'open', 'high', 'low', 'volume']
        )
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')

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
        def _apply_rsi_padding(self, df: pd.DataFrame, period: int) -> None:
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

class OutputFormatter:
    """Handles formatting and writing of output data."""
    
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

class FeatureProcessor:
    """Orchestrates the feature processing workflow."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.df_builder = DataFrameBuilder()
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.output_formatter = OutputFormatter()

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
        # Load and preprocess data
        share_prices: List[SharePrice] = self.data_loader.read_and_reverse_csv(input_file_path)
        
        # Create DataFrame
        df: pd.DataFrame = self.df_builder.create_base_dataframe(share_prices)
        
        # Calculate features
        df = self.indicator_calculator.calculate_features(df, features, custom_params)
        
        # Format and write output
        output_rows: List[str] = self.output_formatter.format_output(df)
        self.output_formatter.write_output(output_rows, output_file_path)

def main():
    processor = FeatureProcessor()
    
    input_path: str = "resources2/APPL/APPL19972007.csv"
    output_path: str = "resources2/output.csv"
    
    # Example of running with custom features and parameters
    custom_params = {
        'rsi': {'periods': range(1, 21)},
        'sma': {'periods': [50, 200]}
    }
    
    processor.run_analysis(
        input_file_path=input_path,
        output_file_path=output_path,
        features=['rsi', 'sma'],
        custom_params=custom_params
    )

if __name__ == "__main__":
    main()
