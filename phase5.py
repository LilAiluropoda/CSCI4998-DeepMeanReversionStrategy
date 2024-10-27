import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from tabulate import tabulate

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

class DataLoader:
    """Handles data loading and preprocessing operations."""
    
    @staticmethod
    def read_csv_file(file_path: str) -> pd.DataFrame:
        """Reads trading data from CSV and returns a pandas DataFrame."""
        df = pd.read_csv(file_path, delimiter=';', header=None)
        df.columns = ['price', 'signal']
        df['price'] = pd.to_numeric(df['price'])
        df['signal'] = pd.to_numeric(df['signal'])
        return df

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
        metrics = {
            "Company": company,
            "Final Return ($)": round(stats.money, 2),
            "Annualized Return (%)": round(((math.exp(math.log(stats.money/10000.0)/number_of_years)-1)*100), 2),
            "Annual Transactions": round(stats.transaction_count / number_of_years, 1),
            "Success Rate (%)": round((stats.success_transaction_count / stats.transaction_count) * 100, 2),
            "Avg Profit per Trade (%)": round((stats.total_percent_profit / stats.transaction_count) * 100, 2),
            "Avg Trade Length (Days)": round(stats.total_transaction_length / stats.transaction_count),
            "Max Profit per Trade (%)": round((stats.maximum_gain / stats.money) * 100, 2),
            "Max Loss per Trade (%)": round((stats.maximum_lost / stats.money) * 100, 2),
            "Maximum Capital ($)": round(stats.maximum_money, 2),
            "Minimum Capital ($)": round(stats.minimum_money, 2),
            "Idle Ratio (%)": round(((data_length - stats.total_transaction_length) / data_length) * 100, 2)
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
        """Saves trading results to a file."""
        results = [[k, v] for k, v in metrics.items()]
        with open("resources2/Results.txt", "w") as writer:
            writer.write("=== Trading Performance Report ===\n")
            writer.write(tabulate(results, headers=["Metric", "Value"], tablefmt="grid"))

class Visualizer:
    """Handles visualization of trading decisions."""

    @staticmethod
    def visualize_trading_decisions(data: pd.DataFrame, company: str) -> None:
        """
        Creates a visualization of trading decisions overlaid on the price chart.
        
        Args:
            data (pd.DataFrame): DataFrame with 'price' and 'signal' columns
            company (str): Stock symbol being analyzed
        """
        # Setup the plot
        dates = range(len(data))
        prices = data['price'].values
        
        plt.figure(figsize=(40,30))
        plt.plot(dates, prices, color='gray', alpha=0.6, label='Price')
        
        # Find and analyze trades
        actual_trades = []
        k = 0
        while k < len(data) - 1:
            if data.loc[k, 'signal'] == 1.0:  # Buy signal
                buy_point = data.loc[k, 'price']
                buy_index = k
                share_number = (10000.0 - 1.0) / buy_point
                force_sell = False
                
                # Look for sell point
                for j in range(k, len(data) - 1):
                    sell_point = data.loc[j, 'price']
                    money_temp = (share_number * sell_point) - 1.0
                    
                    # Check stop loss
                    if 10000.0 * 0.85 > money_temp:
                        actual_trades.append((buy_index, j, buy_point, sell_point, True))
                        k = j + 1
                        break
                    
                    # Check sell signal
                    if data.loc[j, 'signal'] == 2.0 or force_sell:
                        actual_trades.append((buy_index, j, buy_point, sell_point, False))
                        k = j + 1
                        break
                else:
                    k += 1
            else:
                k += 1
        
        # Separate trades by type
        buy_points = [(trade[0], trade[2]) for trade in actual_trades]
        sell_points = [(trade[1], trade[3]) for trade in actual_trades if not trade[4]]
        force_sell_points = [(trade[1], trade[3]) for trade in actual_trades if trade[4]]
        
        # Calculate returns for finding best and worst trades
        trade_returns = [((sell_price - buy_price) / buy_price) * 100 
                        for _, _, buy_price, sell_price, _ in actual_trades]
        best_trade_idx = trade_returns.index(max(trade_returns)) if trade_returns else -1
        worst_trade_idx = trade_returns.index(min(trade_returns)) if trade_returns else -1
        
        # Plot buy points
        if buy_points:
            buy_x, buy_y = zip(*buy_points)
            plt.scatter(buy_x, buy_y, color='green', marker='^', s=500, label='Buy')
        
        # Plot regular sell points
        if sell_points:
            sell_x, sell_y = zip(*sell_points)
            plt.scatter(sell_x, sell_y, color='red', marker='v', s=500, label='Sell')
        
        # Plot force sell points
        if force_sell_points:
            force_x, force_y = zip(*force_sell_points)
            plt.scatter(force_x, force_y, color='black', marker='x', s=500, label='Force Sell')
        
        # Highlight trades and add annotations
        for trade_idx, (buy_idx, sell_idx, buy_price, sell_price, is_force_sell) in enumerate(actual_trades, 1):
            # Highlight holding period
            plt.axvspan(buy_idx, sell_idx, color='blue', alpha=0.1)
            
            # Calculate trade metrics
            profit_pct = ((sell_price - buy_price) / buy_price) * 100
            mid_point = (buy_idx + sell_idx) // 2
            mid_price = max(buy_price, sell_price)
            
            # Prepare annotation text
            if is_force_sell:
                annotation_text = f'Trade {trade_idx}\n{profit_pct:.1f}%\nForce Sell'
            else:
                annotation_text = f'Trade {trade_idx}\n{profit_pct:.1f}%'
            
            # Add special markers for best and worst trades
            if trade_idx - 1 == best_trade_idx:
                plt.plot(sell_idx, sell_price, marker='*', color='gold', markersize=30, 
                        label='Best Trade')
                annotation_text += '\nBEST TRADE'
            elif trade_idx - 1 == worst_trade_idx:
                plt.plot(sell_idx, sell_price, marker='*', color='red', markersize=30, 
                        label='Worst Trade')
                annotation_text += '\nWORST TRADE'
            
            # Add annotation
            plt.annotate(annotation_text, 
                        xy=(mid_point, mid_price),
                        xytext=(0, 30), textcoords='offset points',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Add summary statistics
        total_trades = len(actual_trades)
        profitable_trades = sum(1 for _, _, buy, sell, _ in actual_trades if sell > buy)
        force_sells = sum(1 for trade in actual_trades if trade[4])
        success_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        best_return = max(trade_returns) if trade_returns else 0
        worst_return = min(trade_returns) if trade_returns else 0
        
        summary_text = (
            f'Total Trades: {total_trades}\n'
            f'Profitable Trades: {profitable_trades}\n'
            f'Force Sells: {force_sells}\n'
            f'Success Rate: {success_rate:.1f}%\n'
            f'Best Return: {best_return:.1f}%\n'
            f'Worst Return: {worst_return:.1f}%'
        )
        plt.figtext(0.02, 0.02, summary_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        # Customize plot
        plt.title(f'Trading Decisions for {company}')
        plt.xlabel('Trading Days')
        plt.ylabel('Price')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f'resources2/trading_decisions_{company}.png', bbox_inches='tight', dpi=300)
        plt.close()

class TradingSystem:
    """Main trading system that coordinates all operations for a single company."""

    def __init__(self, company: str):
        self.company = company
        self.data_loader = DataLoader()
        self.backtester = Backtester()
        self.report_generator = BackTestReportGenerator()
        self.visualizer = Visualizer()

    def run(self):
        """Executes the complete trading analysis process."""
        try:
            # Load and process data
            fname: str = "resources2/outputOfTestPrediction.txt"
            data = self.data_loader.read_csv_file(fname)
            
            # Perform backtesting
            stats = self.backtester.process_transactions(data, self.company)
            money_bah = self.backtester.calculate_bah(data)
            
            # Generate and display results
            metrics = self.report_generator.generate_report(
                stats, money_bah, len(data), self.company
            )
            self.report_generator.display_results(metrics)
            self.report_generator.save_results(metrics)
            
            # Visualize trading decisions
            self.visualizer.visualize_trading_decisions(data, self.company)
            
            print(f"\nAnalysis completed successfully for {self.company}")
            print("Results have been saved to 'resources2/Results.txt'")
            print(f"Trading visualization has been saved as 'resources2/trading_decisions_{self.company}.png'")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")

