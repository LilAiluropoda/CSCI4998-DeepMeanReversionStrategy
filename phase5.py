import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from tabulate import tabulate
from phase1 import SMAIndicator
from matplotlib.patches import Patch

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

class online_dc_calculator():
    def __init__(self, threshold: float = 0.1):
        # Data storage
        self.prices = []
        self.time = []
        self.TMV_list = []
        self.T_list = []
        self.colors = []
        self.events = []
        
        # State variables
        self.threshold = threshold
        self.ext_point_n = None
        self.curr_event_max = None
        self.curr_event_min = None
        self.time_point_max = 0
        self.time_point_min = 0
        self.trend_status = 'up'
        self.T = 0
        self.current_index = 0

    def update_point(self, new_price, timestamp=None):
        """
        Process a single new price point and update DC states
        Returns: current event type
        """
        # Initialize if first point
        if self.ext_point_n is None:
            self.ext_point_n = new_price
            self.curr_event_max = new_price
            self.curr_event_min = new_price
            
        # Store price and time
        self.prices.append(new_price)
        if timestamp is None:
            timestamp = self.current_index
        self.time.append(timestamp)
        
        # Calculate TMV
        TMV = (new_price - self.ext_point_n) / (self.ext_point_n * self.threshold)
        self.TMV_list.append(TMV)
        self.T_list.append(self.T)
        self.T += 1

        current_event = None
        
        if self.trend_status == 'up':
            self.colors.append('lime')
            self.events.append('Upward Overshoot')
            current_event = 'Upward Overshoot'

            if new_price < ((1 - self.threshold) * self.curr_event_max):
                self.trend_status = 'down'
                self.curr_event_min = new_price
                self.ext_point_n = self.curr_event_max
                self.T = self.current_index - self.time_point_max

                # Update previous events in current trend
                num_points_change = self.current_index - self.time_point_max
                for j in range(1, num_points_change + 1):
                    self.colors[-j] = 'red'
                    self.events[-j] = 'Downward DCC'
                current_event = 'Downward DCC'
            else:
                if new_price > self.curr_event_max:
                    self.curr_event_max = new_price
                    self.time_point_max = self.current_index
        else:
            self.colors.append('lightcoral')
            self.events.append('Downward Overshoot')
            current_event = 'Downward Overshoot'

            if new_price > ((1 + self.threshold) * self.curr_event_min):
                self.trend_status = 'up'
                self.curr_event_max = new_price
                self.ext_point_n = self.curr_event_min            
                self.T = self.current_index - self.time_point_min

                # Update previous events in current trend
                num_points_change = self.current_index - self.time_point_min
                for j in range(1, num_points_change + 1):
                    self.colors[-j] = 'green'
                    self.events[-j] = 'Upward DCC'
                current_event = 'Upward DCC'
            else:
                if new_price < self.curr_event_min:
                    self.curr_event_min = new_price
                    self.time_point_min = self.current_index

        self.current_index += 1
        return current_event

    def compute_dc_variables(self, prices=None):
        """
        Batch computation of DC variables for a series of prices
        """
        if prices is not None:
            # Reset state for new computation
            self.__init__(self.threshold)
            self.prices = prices
            
        if not self.prices:
            print('Please load the time series data first')
            return
            
        for price in self.prices:
            self.update_point(price)
            
        self.colors = np.array(self.colors)
        print('DC variables computation has finished.')

    def get_current_state(self):
        """
        Return current state information
        """
        return {
            'trend_status': self.trend_status,
            'current_event': self.events[-1] if self.events else None,
            'TMV': self.TMV_list[-1] if self.TMV_list else None,
            'T': self.T,
            'current_price': self.prices[-1] if self.prices else None,
            'ext_point': self.ext_point_n
        }

    def reset(self):
        """
        Reset the calculator state
        """
        self.__init__(self.threshold)

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
    def process_transactions(data: pd.DataFrame) -> TransactionStats:
        """
        Process transactions based on DC trends and signals.
        
        Args:
            data: DataFrame containing price and signal data
        
        Returns:
            TransactionStats: Statistics of processed transactions
        """
        # Process final statistics
        stats = TransactionStats()
        i = 0
        while i < len(data) - 1:
            if data.loc[i, 'signal'] == 1.0:
                i, stats = Backtester.process_transaction(data, i, stats)
            else:
                i += 1
                
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
        results_file = "resources2/Results.csv"
        
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
        # Setup the plot with two subplots sharing x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 30), height_ratios=[3, 1], sharex=True)
        plt.subplots_adjust(hspace=0)

        dates = range(len(data))
        prices = data['price'].values
        
        # Calculate SMAs
        sma_50 = SMAIndicator(prices, 50)
        sma_200 = SMAIndicator(prices, 200)
        
        sma_50_values = [sma_50.calculate(i) for i in range(len(prices))]
        sma_200_values = [sma_200.calculate(i) for i in range(len(prices))]

        # Calculate State
        dc_calculator = online_dc_calculator()
        states = [dc_calculator.update_point(prices[i], i) for i in range(len(prices))]

        # Define colors for different states
        state_colors = {
            'Upward DCC': '#006400',      # Dark green
            'Upward Overshoot': '#90EE90', # Light green
            'Downward DCC': '#8B0000',     # Dark red
            'Downward Overshoot': '#FFB6C1' # Light red
        }

        # Highlight state regions
        current_state = states[0]
        state_start = 0
        
        for i in range(1, len(states)):
            if states[i] != current_state or i == len(states) - 1:
                # Fill the region with appropriate color
                if current_state in state_colors:
                    ax1.axvspan(state_start, i, 
                            color=state_colors[current_state], 
                            alpha=0.2)
                current_state = states[i]
                state_start = i

        # Plot price and SMAs on main chart (ax1)
        ax1.plot(dates, prices, color='gray', alpha=0.6, label='Price')
        ax1.plot(dates, sma_50_values, color='blue', alpha=0.5, label='50-day SMA', linewidth=2)
        ax1.plot(dates, sma_200_values, color='red', alpha=0.5, label='200-day SMA', linewidth=2)

        # Initialize cumulative returns array
        cumulative_returns = np.zeros(len(data))
        
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
                        # Calculate return for this trade and add to cumulative
                        trade_return = ((sell_point - buy_point) / buy_point) * 100
                        cumulative_returns[j:] += trade_return
                        k = j + 1
                        break
                    
                    # Check sell signal
                    if data.loc[j, 'signal'] == 2.0 or force_sell:
                        actual_trades.append((buy_index, j, buy_point, sell_point, False))
                        # Calculate return for this trade and add to cumulative
                        trade_return = ((sell_point - buy_point) / buy_point) * 100
                        cumulative_returns[j:] += trade_return
                        k = j + 1
                        break
                else:
                    k += 1
            else:
                k += 1

        # Plot cumulative returns as bars in the lower subplot (ax2)
        colors = ['green' if x >= 0 else 'red' for x in cumulative_returns]
        ax2.bar(dates, cumulative_returns, color=colors, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Cumulative Return (%)')
        
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
            ax1.scatter(buy_x, buy_y, color='green', marker='^', s=500, label='Buy')
        
        # Plot regular sell points
        if sell_points:
            sell_x, sell_y = zip(*sell_points)
            ax1.scatter(sell_x, sell_y, color='red', marker='v', s=500, label='Sell')
        
        # Plot force sell points
        if force_sell_points:
            force_x, force_y = zip(*force_sell_points)
            ax1.scatter(force_x, force_y, color='black', marker='x', s=500, label='Force Sell')
        
        # Highlight trades
        for trade_idx, (buy_idx, sell_idx, buy_price, sell_price, is_force_sell) in enumerate(actual_trades, 1):
            # Highlight holding period
            ax1.axvspan(buy_idx, sell_idx, color='blue', alpha=0.1)
            
            # Add special markers for best and worst trades
            if trade_idx - 1 == best_trade_idx:
                ax1.plot(sell_idx, sell_price, marker='*', color='gold', markersize=30, 
                        label='Best Trade')
            elif trade_idx - 1 == worst_trade_idx:
                ax1.plot(sell_idx, sell_price, marker='*', color='red', markersize=30, 
                        label='Worst Trade')
        
        # Add summary statistics
        total_trades = len(actual_trades)
        profitable_trades = sum(1 for _, _, buy, sell, _ in actual_trades if sell > buy)
        force_sells = sum(1 for trade in actual_trades if trade[4])
        success_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        best_return = max(trade_returns) if trade_returns else 0
        worst_return = min(trade_returns) if trade_returns else 0
        final_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
        
        summary_text = (
            f'Total Trades: {total_trades}\n'
            f'Profitable Trades: {profitable_trades}\n'
            f'Force Sells: {force_sells}\n'
            f'Success Rate: {success_rate:.1f}%\n'
            f'Best Return: {best_return:.1f}%\n'
            f'Worst Return: {worst_return:.1f}%\n'
            f'Final Return: {final_return:.1f}%'
        )
        plt.figtext(0.02, 0.02, summary_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        # Customize plot
        ax1.set_title(f'Trading Decisions for {company}')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Trading Days')

        # Add legend for states
        state_legend_elements = [
            Patch(facecolor=color, alpha=0.2, label=state)
            for state, color in state_colors.items()
        ]
        ax1.legend(handles=state_legend_elements + ax1.get_legend_handles_labels()[0],
                loc='upper left', bbox_to_anchor=(1, 1))

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
            # Create a copy of the DataFrame to avoid modifying the original
            processed_data = data.copy()
            
            active_position = False
            current_trend = None  # None initially, will be set to 'up' or 'down'

            dc_tracker = online_dc_calculator()
            
            for i in range(len(processed_data)):
                state = dc_tracker.update_point(processed_data.loc[i, 'price'], i)
                
                # Update trend based on DC state
                if state == 'Upward DCC':
                    current_trend = 'up'
                elif state == 'Downward DCC':
                    current_trend = 'down'
                    
                # Get current signal
                current_signal = processed_data.loc[i, 'signal']
                
                # Process signals based on trend
                if current_trend == 'up':
                    if current_signal == 1.0:  # Buy signal
                        active_position = True
                    if current_signal == 2.0:  # Sell signal
                        if active_position == False:
                            processed_data.loc[i, 'signal'] = 0.0  # Cancel the trade
                        else:
                            active_position = False
                elif current_trend == 'down':
                    if active_position:
                        processed_data.loc[i, 'signal'] = 2.0  # Close position
                        active_position = False
                    else:
                        processed_data.loc[i, 'signal'] = 0.0  # No trading in downtrend

            stats = self.backtester.process_transactions(data)
            money_bah = self.backtester.calculate_bah(data)
            
            # Generate and display results
            metrics = self.report_generator.generate_report(
                stats, money_bah, len(data), self.company
            )
            # self.report_generator.display_results(metrics)
            self.report_generator.save_results(metrics)
            
            # Visualize trading decisions
            self.visualizer.visualize_trading_decisions(processed_data, self.company)
            
            print(f"\nAnalysis completed successfully for {self.company}")
            print("Results have been saved to 'resources2/Results.txt'")
            print(f"Trading visualization has been saved as 'resources2/trading_decisions_{self.company}.png'")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")

