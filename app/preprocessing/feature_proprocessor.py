from typing import List, Dict, Callable, Any
import pandas as pd
import numpy as np
from app.preprocessing.indicators.rsi import RSIIndicator
from app.preprocessing.indicators.sma import SMAIndicator
from app.data.datahelper import DataLoader

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
