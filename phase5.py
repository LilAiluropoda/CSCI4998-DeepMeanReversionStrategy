import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from tabulate import tabulate

@dataclass
class TransactionStats:
    """
    Represents statistics for a series of trading transactions.

    Attributes:
        money (float): Current capital amount, initialized at $10,000.
        transaction_count (int): Total number of executed trades.
        success_transaction_count (int): Number of profitable trades.
        failed_transaction_count (int): Number of unprofitable trades.
        total_percent_profit (float): Cumulative percentage profit from all trades.
        maximum_money (float): Highest capital value reached during trading.
        minimum_money (float): Lowest capital value reached during trading.
        maximum_gain (float): Largest profit achieved in a single trade.
        maximum_lost (float): Largest loss incurred in a single trade.
        total_gain (float): Cumulative profit/loss from all trades.
        total_transaction_length (int): Total number of days positions were held.
    """
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

def read_csv_file(file_path: str) -> List[List[str]]:
    """
    Reads a CSV file containing trading data and returns its contents.

    Args:
        file_path (str): Path to the CSV file containing trading data.

    Returns:
        List[List[str]]: A 2D list where each inner list represents a row of trading data.
            Expected format: [price, signal, other_indicators, ...]

    Raises:
        FileNotFoundError: If the specified file doesn't exist.
        csv.Error: If there's an error parsing the CSV file.
    """
    with open(file_path, 'r') as file:
        return list(csv.reader(file, delimiter=';'))

def process_transaction(data: List[List[str]], k: int, stats: TransactionStats) -> Tuple[int, TransactionStats]:
    """
    Processes a single trading transaction from entry to exit point.

    Args:
        data (List[List[str]]): Trading data containing price and signals.
        k (int): Current index in the data representing potential entry point.
        stats (TransactionStats): Current trading statistics object.

    Returns:
        Tuple[int, TransactionStats]: 
            - Next index to process after this transaction
            - Updated trading statistics

    Implementation Details:
        - Calculates entry point based on current price
        - Implements a 15% stop-loss mechanism
        - Monitors for exit signals or forced selling conditions
        - Updates trading statistics upon transaction completion
    """
    buy_point: float = float(data[k][0]) * 100
    share_number: float = (stats.money - 1.0) / buy_point
    force_sell: bool = False

    for j in range(k, len(data) - 1):
        sell_point: float = float(data[j][0]) * 100
        money_temp: float = (share_number * sell_point) - 1.0
        
        if stats.money * 0.85 > money_temp:
            stats.money = money_temp
            force_sell = True
        
        if float(data[j][1]) == 2.0 or force_sell:
            sell_point = float(data[j][0]) * 100
            gain: float = sell_point - buy_point
            
            stats = update_stats(stats, gain, share_number, sell_point, j - k)
            return j + 1, stats
    
    return k + 1, stats

def update_stats(stats: TransactionStats, gain: float, share_number: float, 
                sell_point: float, transaction_length: int) -> TransactionStats:
    """
    Updates trading statistics after completing a transaction.

    Args:
        stats (TransactionStats): Current trading statistics object.
        gain (float): Profit/loss from the current transaction.
        share_number (float): Number of shares traded.
        sell_point (float): Exit price of the transaction.
        transaction_length (int): Duration of the trade in days.

    Returns:
        TransactionStats: Updated statistics object reflecting the latest transaction.

    Updates:
        - Success/failure counts
        - Maximum gain/loss records
        - Capital high/low watermarks
        - Cumulative statistics (total gain, transaction length, etc.)
    """
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

def process_transactions(data: List[List[str]], company: str) -> TransactionStats:
    """
    Processes all trading transactions for a given company.

    Args:
        data (List[List[str]]): Complete trading data for the company.
        company (str): Stock symbol or identifier of the company.

    Returns:
        TransactionStats: Compiled statistics for all transactions.

    Process Flow:
        1. Initializes fresh statistics object
        2. Iterates through data looking for entry signals (1.0)
        3. Processes each identified trading opportunity
        4. Accumulates statistics across all transactions
    """
    stats: TransactionStats = TransactionStats()
    
    k: int = 0
    while k < len(data) - 1:
        if float(data[k][1]) == 1.0:
            k, stats = process_transaction(data, k, stats)
        else:
            k += 1
    
    return stats

def calculate_bah(data: List[List[str]]) -> float:
    """
    Calculates the result of a Buy-and-Hold strategy.

    Args:
        data (List[List[str]]): Complete trading data from start to end.

    Returns:
        float: Final capital after implementing Buy-and-Hold strategy.

    Calculation:
        - Initial investment: $10,000
        - Buys at first available price
        - Holds until the last data point
        - Accounts for transaction fee of $1
    """
    money_bah: float = 10000.0
    buy_point_bah: float = float(data[0][0])
    share_number_bah: float = (money_bah - 1.0) / buy_point_bah
    return (float(data[-1][0]) * share_number_bah) - 1.0

def generate_report(stats: TransactionStats, money_bah: float, data_length: int, company: str) -> List[Any]:
    """
    Generates a comprehensive performance report for a trading strategy.

    Args:
        stats (TransactionStats): Compiled trading statistics.
        money_bah (float): Result from Buy-and-Hold strategy.
        data_length (int): Total number of trading days.
        company (str): Stock symbol being analyzed.

    Returns:
        List[Any]: Formatted list of performance metrics including:
            - Company identifier
            - Final capital
            - Annualized return
            - Transaction frequency
            - Success rate
            - Average profit per trade
            - Average trade duration
            - Maximum profit/loss percentages
            - Capital range
            - Market participation rate

    Calculations:
        - Annualized metrics are based on 365 days per year
        - Percentages are rounded to 2 decimal places
        - Monetary values are rounded to 2 decimal places
    """
    number_of_years: float = (data_length - 1) / 365
    annualized_return = ((math.exp(math.log(stats.money/10000.0)/number_of_years)-1)*100)
    ann_num_transactions = stats.transaction_count / number_of_years
    percent_success = (stats.success_transaction_count / stats.transaction_count) * 100
    avg_profit_per_transaction = (stats.total_percent_profit / stats.transaction_count) * 100
    avg_transaction_length = stats.total_transaction_length / stats.transaction_count
    max_profit_percent = (stats.maximum_gain / stats.money) * 100
    max_loss_percent = (stats.maximum_lost / stats.money) * 100
    idle_ratio = ((data_length - stats.total_transaction_length) / data_length) * 100

    return [
        company,
        round(stats.money, 2),
        round(annualized_return, 2),
        round(ann_num_transactions, 1),
        round(percent_success, 2),
        round(avg_profit_per_transaction, 2),
        round(avg_transaction_length),
        round(max_profit_percent, 2),
        round(max_loss_percent, 2),
        round(stats.maximum_money, 2),
        round(stats.minimum_money, 2),
        round(idle_ratio, 2)
    ]

def display_results(results: List[List[Any]]):
    """
    Displays trading results in a formatted grid layout.

    Args:
        results (List[List[Any]]): List of performance metrics for each company.

    Output Format:
        Displays a grid with columns for:
        - Company identifier
        - GA Return (strategy performance in dollars)
        - Return percentage (annualized)
        - Transaction frequency
        - Success rate
        - Average profit per trade
        - Average trade duration
        - Maximum profit/loss percentages
        - Capital extremes
        - Idle ratio

    Note:
        Uses tabulate library for consistent and readable formatting
    """
    headers = [
        "Company", "GA Return ($)", "Rtn % (%)", "Ann.#ofT (Deals)", "Success % (%)", 
        "ApT (%)", "L (Days)", "MpT (%)", "MLT (%)", "MxC ($)", "MinC ($)", "IR (%)"
    ]
    print(tabulate(results, headers=headers, tablefmt="grid"))

def visualize_trading_decisions(data: List[List[str]], company: str) -> None:
    """
    Creates a visualization of actual trading decisions overlaid on the price chart.

    Args:
        data (List[List[str]]): Trading data containing prices and signals.
        company (str): Stock symbol being analyzed.

    Plots:
        - Price line chart
        - Actual buy points (green markers)
        - Actual sell points (red markers)
        - Force sell points (black markers)
        - Best trade (gold star)
        - Worst trade (red star)
        - Actual holding periods (blue shaded areas)
        - Profit/Loss annotations for each trade
    """
    # Extract price data
    dates = list(range(len(data)))
    prices = [float(row[0]) for row in data]

    # Create figure and axis
    plt.figure(figsize=(40,30))
    plt.plot(dates, prices, color='gray', alpha=0.6, label='Price')

    # Simulate actual transactions
    actual_trades = []  # List to store (buy_index, sell_index, buy_price, sell_price, is_force_sell)
    k = 0
    while k < len(data) - 1:
        if float(data[k][1]) == 1.0:  # Buy signal
            buy_point = float(data[k][0])
            buy_index = k
            share_number = (10000.0 - 1.0) / buy_point
            force_sell = False

            # Look for sell point
            for j in range(k, len(data) - 1):
                sell_point = float(data[j][0])
                money_temp = (share_number * sell_point) - 1.0

                # Check stop loss
                if 10000.0 * 0.85 > money_temp:
                    actual_trades.append((buy_index, j, buy_point, sell_point, True))
                    k = j + 1
                    break

                # Check sell signal or force sell
                if float(data[j][1]) == 2.0 or force_sell:
                    actual_trades.append((buy_index, j, buy_point, sell_point, False))
                    k = j + 1
                    break
            else:
                k += 1
        else:
            k += 1

    # Separate trades by type
    buy_points = [(trade[0], trade[2]) for trade in actual_trades]
    sell_points = [(trade[1], trade[3]) for trade in actual_trades if not trade[4]]  # Regular sells
    force_sell_points = [(trade[1], trade[3]) for trade in actual_trades if trade[4]]  # Force sells

    # Find best and worst trades
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

    # Highlight actual holding periods and annotate profits
    for trade_idx, (buy_idx, sell_idx, buy_price, sell_price, is_force_sell) in enumerate(actual_trades, 1):
        # Highlight holding period
        plt.axvspan(buy_idx, sell_idx, color='blue', alpha=0.1)

        # Calculate profit/loss
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

        # Annotate trade
        plt.annotate(annotation_text, 
                    xy=(mid_point, mid_price),
                    xytext=(0, 30), textcoords='offset points',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Customize plot
    plt.title(f'Actual Trading Decisions for {company}')
    plt.xlabel('Trading Days')
    plt.ylabel('Price')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)

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

    plt.tight_layout()
    plt.savefig(f'resources2/trading_decisions_{company}.png', bbox_inches='tight', dpi=300)
    plt.close()


def phase_process() -> None:
    """
    Main workflow function with added visualization.
    """
    fname: str = "resources2/outputOfTestPrediction.txt"
    companies = ["WMT"]
    results = []

    for company in companies:
        # Read and process data
        data: List[List[str]] = read_csv_file(fname)
        stats: TransactionStats = process_transactions(data, company)
        money_bah: float = calculate_bah(data)
        result = generate_report(stats, money_bah, len(data), company)
        results.append(result)

        # Create visualization
        # visualize_trading_decisions(data, company)

    # Display results table
    display_results(results)

    # Write results to file
    with open("resources2/Results.txt", "w") as writer:
        writer.write(tabulate(results, headers=[
            "Company", "GA + MLP Return ($)", "Rtn % (%)", "Ann.#ofT (Deals)", "Success % (%)", 
            "ApT (%)", "L (Days)", "MpT (%)", "MLT (%)", "MxC ($)", "MinC ($)", "IR (%)"
        ], tablefmt="grid"))

if __name__ == "__main__":
    phase_process()
