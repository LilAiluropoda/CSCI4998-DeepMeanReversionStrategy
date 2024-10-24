import csv
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from tabulate import tabulate

@dataclass
class TransactionStats:
    """Represents statistics for a series of transactions."""
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
    """Reads a CSV file and returns its contents as a list of lists."""
    with open(file_path, 'r') as file:
        return list(csv.reader(file, delimiter=';'))

def process_transaction(data: List[List[str]], k: int, stats: TransactionStats) -> Tuple[int, TransactionStats]:
    """Processes a single transaction and updates the statistics."""
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
    """Updates transaction statistics based on the current trade."""
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
    """Processes all transactions in the data for a given company."""
    stats: TransactionStats = TransactionStats()
    
    k: int = 0
    while k < len(data) - 1:
        if float(data[k][1]) == 1.0:
            k, stats = process_transaction(data, k, stats)
        else:
            k += 1
    
    return stats

def calculate_bah(data: List[List[str]]) -> float:
    """Calculates the Buy and Hold (BaH) strategy result."""
    money_bah: float = 10000.0
    buy_point_bah: float = float(data[0][0])
    share_number_bah: float = (money_bah - 1.0) / buy_point_bah
    return (float(data[-1][0]) * share_number_bah) - 1.0

def generate_report(stats: TransactionStats, money_bah: float, data_length: int, company: str) -> List[Any]:
    """Generates a report row for the grid view."""
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
    """Displays results in a grid format."""
    headers = [
        "Company", "GA Return ($)", "Rtn % (%)", "Ann.#ofT (Deals)", "Success % (%)", 
        "ApT (%)", "L (Days)", "MpT (%)", "MLT (%)", "MxC ($)", "MinC ($)", "IR (%)"
    ]
    print(tabulate(results, headers=headers, tablefmt="grid"))

def phase_process() -> None:
    """Main process function that handles the entire workflow."""
    fname: str = "resources2/outputOfTestPrediction.txt"
    companies = ["WMT"]
    results = []

    for company in companies:
        data: List[List[str]] = read_csv_file(fname)
        stats: TransactionStats = process_transactions(data, company)
        money_bah: float = calculate_bah(data)
        result = generate_report(stats, money_bah, len(data), company)
        results.append(result)

    display_results(results)

    # Optional: Write detailed results to file
    with open("resources2/Results.txt", "w") as writer:
        writer.write(tabulate(results, headers=[
            "Company", "GA + MLP Return ($)", "Rtn % (%)", "Ann.#ofT (Deals)", "Success % (%)", 
            "ApT (%)", "L (Days)", "MpT (%)", "MLT (%)", "MxC ($)", "MinC ($)", "IR (%)"
        ], tablefmt="grid"))

if __name__ == "__main__":
    phase_process()
