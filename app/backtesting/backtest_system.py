import numpy as np
import pandas as pd
import math
import os
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from tabulate import tabulate
from pathlib import Path
from app.utils.path_config import PathConfig
from app.visualization.trade_decision import TradeDecisionVisualizer


@dataclass
class TransactionStats:
    """Represents statistics for a series of trading transactions."""
    money: float = 10000.0
    transaction_count: int = 0
    success_transaction_count: int = 0
    failed_transaction_count: int = 0
    total_percent_profit: float = 0.0
    maximum_money: float = 0.0
    minimum_money: float = 10000.0
    maximum_gain: float = 0.0
    maximum_lost: float = 100.0
    total_gain: float = 0.0
    total_transaction_length: int = 0

class Backtester:
    """Handles backtesting operations and calculations."""

    @staticmethod
    def process_transaction(data: pd.DataFrame, k: int, stats: TransactionStats) -> Tuple[int, TransactionStats]:
        buy_point: float = data.loc[k, 'price'] * 100
        share_number: float = (stats.money - 1.0) / buy_point
        force_sell: bool = False

        for j in range(k, len(data) - 1):
            sell_point: float = data.loc[j, 'price'] * 100
            money_temp: float = (share_number * sell_point) - 1.0
            
            if stats.money * 0.85 > money_temp:
                stats.money = money_temp
                force_sell = True
            
            if data.loc[j, 'signal'] == 2.0 or force_sell:
                sell_point = data.loc[j, 'price'] * 100
                gain: float = sell_point - buy_point
                stats = Backtester._update_stats(stats, gain, share_number, sell_point, j - k)
                return j + 1, stats
        
        return k + 1, stats

    @staticmethod
    def _update_stats(stats: TransactionStats, gain: float, share_number: float, 
                    sell_point: float, transaction_length: int) -> TransactionStats:
        """Update transaction statistics after a trade"""
        stats.success_transaction_count += 1 if gain > 0 else 0
        stats.failed_transaction_count += 1 if gain <= 0 else 0
        stats.maximum_gain = max(stats.maximum_gain, gain)
        stats.maximum_lost = min(stats.maximum_lost, gain)
        stats.money = (share_number * sell_point) - 1.0
        stats.maximum_money = max(stats.maximum_money, stats.money)
        stats.minimum_money = min(stats.minimum_money, stats.money)
        stats.transaction_count += 1
        stats.total_percent_profit += gain / sell_point
        stats.total_transaction_length += transaction_length
        stats.total_gain += gain
        return stats

    def analyze_transactions(self, data: pd.DataFrame) -> Tuple[TransactionStats, float]:
        """Perform complete transaction analysis"""
        stats = TransactionStats()
        k: int = 0
        while k < len(data) - 1:
            if data.loc[k, 'signal'] == 1.0:
                k, stats = self.process_transaction(data, k, stats)
            else:
                k += 1
                
        money_bah = self._calculate_bah(data)
        return stats, money_bah

    @staticmethod
    def _calculate_bah(data: pd.DataFrame) -> float:
        """Calculate Buy and Hold strategy results"""
        money_bah: float = 10000.0
        buy_point_bah: float = data.loc[0, 'price']
        share_number_bah: float = (money_bah - 1.0) / buy_point_bah
        return (data.loc[len(data)-1, 'price'] * share_number_bah) - 1.0

class PerformanceMetricsCalculator:
    """Calculates trading performance metrics"""
    
    @staticmethod
    def calculate_metrics(stats: TransactionStats, money_bah: float, 
                         data_length: int, company: str) -> Dict[str, float]:
        """Calculate comprehensive trading metrics"""
        number_of_years: float = (data_length - 1) / 365
        
        # Calculate base metrics
        strategy_return_pct = (stats.money / 10000.0 - 1) * 100
        strategy_annual_return = ((math.exp(math.log(stats.money/10000.0)/number_of_years)-1)*100)
        
        bah_return_pct = (money_bah / 10000.0 - 1) * 100
        bah_annual_return = ((math.exp(math.log(money_bah/10000.0)/number_of_years)-1)*100)
        
        return {
            "Company": company,
            "Final_Return": round(stats.money, 2),
            "Final_Return_Pct": round(strategy_return_pct, 2),
            "Annualized_Return": round(strategy_annual_return, 2),
            "BaH_Final_Return": round(money_bah, 2),
            "BaH_Return_Pct": round(bah_return_pct, 2),
            "BaH_Annualized_Return": round(bah_annual_return, 2),
            "Strategy_Outperformance": round(strategy_return_pct - bah_return_pct, 2),
            "Annual_Outperformance": round(strategy_annual_return - bah_annual_return, 2),
            "Annual_Transactions": round(stats.transaction_count / number_of_years, 1),
            "Success_Rate": round((stats.success_transaction_count / stats.transaction_count) * 100, 2),
            "Avg_Profit_per_Trade": round((stats.total_percent_profit / stats.transaction_count) * 100, 2),
            "Avg_Trade_Length": round(stats.total_transaction_length / stats.transaction_count),
            "Max_Profit_per_Trade": round((stats.maximum_gain / stats.money) * 100, 2),
            "Max_Loss_per_Trade": round((stats.maximum_lost / stats.money) * 100, 2),
            "Maximum_Capital": round(stats.maximum_money, 2),
            "Minimum_Capital": round(stats.minimum_money, 2),
            "Idle_Ratio": round(((data_length - stats.total_transaction_length) / data_length) * 100, 2)
        }

class ResultsManager:
    """Manages the storage and display of trading results"""
    
    @staticmethod
    def save_results(metrics: Dict[str, float]) -> None:
        """Save results to CSV file"""
        results_path = PathConfig.RESULTS_FILE
        df_new = pd.DataFrame([metrics])
        
        try:
            if results_path.exists():
                df_existing = pd.read_csv(results_path)
                # Check if entry already exists
                mask = (df_existing['Company'] == metrics['Company']) & \
                      (df_existing['Year'] == metrics['Year'])
                
                if mask.any():
                    # Update existing entry
                    df_existing.loc[mask] = df_new.iloc[0]
                    df_combined = df_existing
                else:
                    # Add new entry
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new
            
            # Sort by Company and Year
            df_combined = df_combined.sort_values(['Company', 'Year'])
            
            df_combined.to_csv(results_path, index=False)
            print(f"Results saved to {results_path} for {metrics['Company']} - Year {metrics['Year']}")
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")

    @staticmethod
    def display_results(metrics: Dict[str, float]) -> None:
        """Display results in formatted table"""
        results = [[k, v] for k, v in metrics.items()]
        print(f"\n=== Trading Performance Report - {metrics['Company']} ({metrics['Year']}) ===")
        print(tabulate(results, headers=["Metric", "Value"], tablefmt="grid"))

class TradingSystem:
    """Main trading system coordinator"""

    def __init__(self, company: str, year: int):
        self.company = company
        self.year = year
        self.backtester = Backtester()
        self.metrics_calculator = PerformanceMetricsCalculator()
        self.results_manager = ResultsManager()
        self.visualizer = TradeDecisionVisualizer()

    def run(self) -> None:
        """Execute complete trading analysis"""
        try:
            # Load data
            data = pd.read_csv(
                PathConfig.OUTPUT_TEST_PREDICTION, 
                delimiter=';', 
                header=None,
                names=['price', 'signal']
            )
            
            # Perform analysis
            stats, money_bah = self.backtester.analyze_transactions(data)
            
            # Calculate and save metrics with year information
            metrics = self.metrics_calculator.calculate_metrics(
                stats, money_bah, len(data), self.company
            )
            
            # Add year information to metrics
            metrics.update({
                "Year": self.year,
                "Training_Period": f"{self.year}-{self.year+3}",
                "Testing_Period": f"{self.year+4}"
            })
            
            # Save results
            self.results_manager.save_results(metrics)
            
            # Generate visualization (commented out as in original)
            # self.visualizer.visualize_trading_decisions(data, self.company)
            
            print(f"\nAnalysis completed for {self.company} - Year {self.year}")
            
        except Exception as e:
            print(f"Error during analysis for {self.company} - Year {self.year}: {str(e)}")
