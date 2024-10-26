from typing import List
import csv
import pandas as pd
import numpy as np
from rsi import RSIIndicator
from sma import SMAIndicator

WINDOW_SIZE: int = 11  # Sh 6

class SharePrice:
    """Represents a single day's stock price data."""

    def __init__(self, date: str, open_price: float, high: float, low: float, close: float, volume: float, adj_close: float):
        self.date: str = date
        self.open: float = open_price
        self.high: float = high
        self.low: float = low
        self.close: float = close
        self.volume: float = volume
        self.adj_close: float = adj_close

def read_csv(file_path: str) -> List[SharePrice]:
    """
    Reads stock price data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        List[SharePrice]: A list of SharePrice objects.
    """
    share_prices: List[SharePrice] = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                date: str = row[0]
                adj_close: float = float(row[6])
                adj_factor: float = adj_close / float(row[4])
                
                share_price = SharePrice(
                    date=date,
                    open_price=float(row[1]) * adj_factor,
                    high=float(row[2]) * adj_factor,
                    low=float(row[3]) * adj_factor,
                    close=adj_close,
                    volume=float(row[5]),
                    adj_close=adj_close
                )
                share_prices.append(share_price)
            except (ValueError, IndexError) as e:
                print(f"Skipping row due to error: {e}")
    return share_prices

def create_base_dataframe(share_prices: List[SharePrice]) -> pd.DataFrame:
    """
    Creates and initializes the base DataFrame from share price data.

    Args:
        share_prices (List[SharePrice]): List of SharePrice objects.

    Returns:
        pd.DataFrame: Initial DataFrame with basic price data.
    """
    df: pd.DataFrame = pd.DataFrame(
        [(sp.date, sp.close, sp.open, sp.high, sp.low, sp.volume) 
         for sp in share_prices],
        columns=['date', 'close', 'open', 'high', 'low', 'volume']
    )
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')

def apply_rsi_padding(df: pd.DataFrame, period: int) -> None:
    """
    Applies custom padding to RSI values for a specific period.

    Args:
        df (pd.DataFrame): DataFrame containing RSI values.
        period (int): RSI period to pad.
    """
    df.loc[0, f'rsi{period}'] = 0.0
    df.loc[1, f'rsi{period}'] = 100.0
    
    for j in range(2, len(df)):
        if j < period:
            df.loc[j, f'rsi{period}'] = df.loc[j, f'rsi{j}']

def calculate_rsi_indicators(df: pd.DataFrame, close_prices: np.ndarray) -> pd.DataFrame:
    """
    Calculates RSI indicators for periods 1-20 with padding.

    Args:
        df (pd.DataFrame): Input DataFrame.
        close_prices (np.ndarray): Array of closing prices.

    Returns:
        pd.DataFrame: DataFrame with RSI indicators added.
    """
    for period in range(1, 21):
        rsi_indicator: RSIIndicator = RSIIndicator(close_prices, period)
        df[f'rsi{period}'] = [rsi_indicator.calculate(j) for j in range(len(df))]
        apply_rsi_padding(df, period)
    return df

def calculate_sma_indicators(df: pd.DataFrame, close_prices: np.ndarray) -> pd.DataFrame:
    """
    Calculates SMA indicators for 50 and 200 periods.

    Args:
        df (pd.DataFrame): Input DataFrame.
        close_prices (np.ndarray): Array of closing prices.

    Returns:
        pd.DataFrame: DataFrame with SMA indicators added.
    """
    sma_periods = [50, 200]
    for period in sma_periods:
        sma_indicator: SMAIndicator = SMAIndicator(close_prices, period)
        df[f'sma{period}'] = [sma_indicator.calculate(j) for j in range(len(df))]
    return df

def process_data(share_prices: List[SharePrice]) -> pd.DataFrame:
    """
    Processes share price data to calculate RSI and SMA indicators with specific padding.

    Args:
        share_prices (List[SharePrice]): List of SharePrice objects.

    Returns:
        pd.DataFrame: Processed data with calculated indicators and padding.
    """
    # Initialize base DataFrame
    df: pd.DataFrame = create_base_dataframe(share_prices)
    close_prices: np.ndarray = df['close'].values
    
    # Calculate technical indicators
    df = calculate_rsi_indicators(df, close_prices)
    df = calculate_sma_indicators(df, close_prices)
    
    return df

def run_phase1() -> None:   
    """
    Runs the first phase of data processing and writes results to a CSV file.
    """
    share_prices: List[SharePrice] = read_csv("resources2/reverseFile.csv")
    df: pd.DataFrame = process_data(share_prices)
    
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
    
    with open("resources2/output.csv", "w", newline='') as f:
        for row in output_rows:
            f.write(f"{row}\n")

if __name__ == "__main__":
    run_phase1()
