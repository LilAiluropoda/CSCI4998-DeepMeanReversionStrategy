from typing import List
import csv
from datetime import datetime
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator

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

class CumulatedGainsIndicator:
    def __init__(self, indicator, timeFrame):
        self.indicator = indicator
        self.timeFrame = timeFrame

    def getValue(self, index):
        sumOfGains = 0
        for i in range(max(1, index - self.timeFrame + 1), index + 1):
            if self.indicator[i] > self.indicator[i - 1]:
                sumOfGains += self.indicator[i] - self.indicator[i - 1]
        return sumOfGains

class CumulatedLossesIndicator:
    def __init__(self, indicator, timeFrame):
        self.indicator = indicator
        self.timeFrame = timeFrame

    def getValue(self, index):
        sumOfLosses = 0
        for i in range(max(1, index - self.timeFrame + 1), index + 1):
            if self.indicator[i] < self.indicator[i - 1]:
                sumOfLosses += self.indicator[i - 1] - self.indicator[i]
        return sumOfLosses

class AverageGainIndicator:
    def __init__(self, indicator, timeFrame):
        self.cumulatedGains = CumulatedGainsIndicator(indicator, timeFrame)
        self.timeFrame = timeFrame

    def getValue(self, index):
        realTimeFrame = min(self.timeFrame, index + 1)
        return self.cumulatedGains.getValue(index) / realTimeFrame

class AverageLossIndicator:
    def __init__(self, indicator, timeFrame):
        self.cumulatedLosses = CumulatedLossesIndicator(indicator, timeFrame)
        self.timeFrame = timeFrame

    def getValue(self, index):
        realTimeFrame = min(self.timeFrame, index + 1)
        return self.cumulatedLosses.getValue(index) / realTimeFrame

class RSIIndicator:
    def __init__(self, indicator, timeFrame):
        self.indicator = indicator
        self.timeFrame = timeFrame
        self.averageGainIndicator = AverageGainIndicator(indicator, timeFrame)
        self.averageLossIndicator = AverageLossIndicator(indicator, timeFrame)

    def calculate(self, index):
        if index == 0:
            return 0

        averageLoss = self.averageLossIndicator.getValue(index)
        if averageLoss == 0:
            return 100

        averageGain = self.averageGainIndicator.getValue(index)
        relativeStrength = averageGain / averageLoss if averageLoss != 0 else float('inf')

        ratio = 100 / (1 + relativeStrength)
        return 100 - ratio

def process_data(share_prices: List[SharePrice]) -> pd.DataFrame:
    """
    Processes share price data to calculate RSI and SMA indicators with specific padding.

    Args:
        share_prices (List[SharePrice]): List of SharePrice objects.

    Returns:
        pd.DataFrame: Processed data with calculated indicators and padding.
    """
    df: pd.DataFrame = pd.DataFrame([(sp.date, sp.close, sp.open, sp.high, sp.low, sp.volume) for sp in share_prices], 
                                    columns=['date', 'close', 'open', 'high', 'low', 'volume'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    close_prices = df['close'].values
    
    # Calculate RSI for different periods with custom padding
    for i in range(1, 21):
        rsi_indicator = RSIIndicator(close_prices, i)
        df[f'rsi{i}'] = [rsi_indicator.calculate(j) for j in range(len(df))]
        
        # Custom padding
        df.loc[0, f'rsi{i}'] = 0.0
        df.loc[1, f'rsi{i}'] = 100.0
        
        for j in range(2, len(df)):
            if j < i:
                df.loc[j, f'rsi{i}'] = df.loc[j, f'rsi{j}']
    
    # Calculate SMA with simple padding
    df['sma50'] = df['close'].rolling(window=50, min_periods=1).mean()
    df['sma200'] = df['close'].rolling(window=200, min_periods=1).mean()
    
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
            *[f"{row[f'rsi{i}']:.1f}" for i in range(1, 21)],
            f"{row['sma50']:.2f}",
            f"{row['sma200']:.2f}"
        ]
        output_rows.append(';'.join(output_row))
        print(f"counterRow: {index + 1}")
    
    with open("resources2/output.csv", "w", newline='') as f:
        for row in output_rows:
            f.write(f"{row}\n")


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
