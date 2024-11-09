import numpy as np
import pandas as pd
import math
import os
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from tabulate import tabulate
from app.data.datahelper import DataLoader
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
                stats = Backtester.update_stats(stats, gain, share_number, sell_point, j - k)
                return j + 1, stats
        
        return k + 1, stats

    @staticmethod
    def update_stats(stats: TransactionStats, gain: float, share_number: float, 
                    sell_point: float, transaction_length: int) -> TransactionStats:
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

    @staticmethod
    def process_transactions(data: pd.DataFrame, company: str) -> TransactionStats:
        stats: TransactionStats = TransactionStats()
        k: int = 0
        while k < len(data) - 1:
            if data.loc[k, 'signal'] == 1.0:
                k, stats = Backtester.process_transaction(data, k, stats)
            else:
                k += 1
        return stats

    @staticmethod
    def calculate_bah(data: pd.DataFrame) -> float:
        money_bah: float = 10000.0
        buy_point_bah: float = data.loc[0, 'price']
        share_number_bah: float = (money_bah - 1.0) / buy_point_bah
        return (data.loc[len(data)-1, 'price'] * share_number_bah) - 1.0

class BackTestReportGenerator:
    """Handles report generation and result display for a single company."""

    @staticmethod
    def generate_report(stats: TransactionStats, money_bah: float, data_length: int, company: str) -> Dict[str, float]:
        """
        Generates a comprehensive report for a single company's trading performance.
        
        Returns:
            Dict containing all calculated metrics
        """
        number_of_years: float = (data_length - 1) / 365
        
        # Calculate BaH metrics
        bah_return = money_bah - 10000.0
        bah_return_pct = (money_bah / 10000.0 - 1) * 100
        bah_annual_return = ((math.exp(math.log(money_bah/10000.0)/number_of_years)-1)*100)
        
        # Calculate strategy metrics
        strategy_return_pct = (stats.money / 10000.0 - 1) * 100
        strategy_annual_return = ((math.exp(math.log(stats.money/10000.0)/number_of_years)-1)*100)
        outperformance = strategy_return_pct - bah_return_pct
        annual_outperformance = strategy_annual_return - bah_annual_return
        
        metrics = {
            "Company": company,
            "Final_Return": round(stats.money, 2),
            "Final_Return_Pct": round(strategy_return_pct, 2),
            "Annualized_Return": round(strategy_annual_return, 2),
            "BaH_Final_Return": round(money_bah, 2),
            "BaH_Return_Pct": round(bah_return_pct, 2),
            "BaH_Annualized_Return": round(bah_annual_return, 2),
            "Strategy_Outperformance": round(outperformance, 2),
            "Annual_Outperformance": round(annual_outperformance, 2),
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
        return metrics

    @staticmethod
    def display_results(metrics: Dict[str, float]):
        """Displays trading results in a formatted table."""
        results = [[k, v] for k, v in metrics.items()]
        print("\n=== Trading Performance Report ===")
        print(tabulate(results, headers=["Metric", "Value"], tablefmt="grid"))

    @staticmethod
    def save_results(metrics: Dict[str, float]):
        """Saves trading results to a CSV file, appending if file exists."""
        results_file = "C:\\Users\\Steve\\Desktop\\Projects\\fyp\\app\\data\\stock_data\\Results.csv"
        
        # Convert metrics to DataFrame
        df_new = pd.DataFrame([metrics])
        
        try:
            # Try to read existing CSV file
            if os.path.exists(results_file):
                df_existing = pd.read_csv(results_file)
                # Append new results
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new
            
            # Save to CSV
            df_combined.to_csv(results_file, index=False)
            print(f"Results appended to {results_file}")
            
        except Exception as e:
            print(f"Error saving results to CSV: {str(e)}")

class TradingSystem:
    """Main trading system that coordinates all operations for a single company."""

    def __init__(self, company: str):
        self.company = company
        self.data_loader = DataLoader()
        self.backtester = Backtester()
        self.report_generator = BackTestReportGenerator()
        self.visualizer = TradeDecisionVisualizer()

    def run(self):
        """Executes the complete trading analysis process."""
        try:
            # Load and process data
            fname: str = "C:\\Users\\Steve\\Desktop\\Projects\\fyp\\app\\data\\stock_data\\outputOfTestPrediction.txt"
            data = self.data_loader.load_signal_data(fname)
            
            # Perform backtesting
            stats = self.backtester.process_transactions(data, self.company)
            money_bah = self.backtester.calculate_bah(data)
            
            # Generate and display results
            metrics = self.report_generator.generate_report(
                stats, money_bah, len(data), self.company
            )
            # self.report_generator.display_results(metrics)
            self.report_generator.save_results(metrics)
            
            # Visualize trading decisions
            self.visualizer.visualize_trading_decisions(data, self.company)
            
            print(f"\nAnalysis completed successfully for {self.company}")
            print("Results have been saved to 'data/stock_data/Results.txt'")
            print(f"Trading visualization has been saved as 'data/plots/trading_decisions/trading_decisions_{self.company}.png'")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")

