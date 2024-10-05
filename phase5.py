import csv
import math
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class TransactionStats:
    money: float
    transaction_count: int
    success_transaction_count: int
    failed_transaction_count: int
    total_percent_profit: float
    maximum_money: float
    minimum_money: float
    maximum_gain: float
    maximum_lost: float
    total_gain: float
    total_transaction_length: int

def read_csv_file(file_path: str) -> List[List[str]]:
    with open(file_path, 'r') as file:
        return list(csv.reader(file, delimiter=';'))

def process_transaction(data: List[List[str]], k: int, stats: TransactionStats) -> Tuple[int, TransactionStats]:
    buy_point = float(data[k][0]) * 100
    share_number = (stats.money - 1.0) / buy_point
    force_sell = False

    for j in range(k, len(data) - 1):
        sell_point = float(data[j][0]) * 100
        money_temp = (share_number * sell_point) - 1.0
        
        if stats.money * 0.85 > money_temp:
            stats.money = money_temp
            force_sell = True
        
        if float(data[j][1]) == 2.0 or force_sell:
            sell_point = float(data[j][0]) * 100
            gain = sell_point - buy_point
            
            stats.success_transaction_count += 1 if gain > 0 else 0
            stats.failed_transaction_count += 1 if gain <= 0 else 0
            stats.maximum_gain = max(stats.maximum_gain, gain)
            stats.maximum_lost = min(stats.maximum_lost, gain)
            stats.money = (share_number * sell_point) - 1.0
            stats.maximum_money = max(stats.maximum_money, stats.money)
            stats.minimum_money = min(stats.minimum_money, stats.money)
            stats.transaction_count += 1
            stats.total_percent_profit += gain / buy_point
            stats.total_transaction_length += j - k
            stats.total_gain += gain
            
            return j + 1, stats
    
    return k + 1, stats

def process_transactions(data: List[List[str]]) -> TransactionStats:
    stats = TransactionStats(10000.0, 0, 0, 0, 0.0, 0.0, 10000.0, 0.0, 100.0, 0.0, 0)
    
    k = 0
    while k < len(data) - 1:
        if float(data[k][1]) == 1.0:
            k, stats = process_transaction(data, k, stats)
        else:
            k += 1
    
    return stats

def calculate_bah(data: List[List[str]]) -> float:
    money_bah = 10000.0
    buy_point_bah = float(data[0][0])
    share_number_bah = (money_bah - 1.0) / buy_point_bah
    return (float(data[-1][0]) * share_number_bah) - 1.0

def generate_report(stats: TransactionStats, money_bah: float, data_length: int) -> List[str]:
    number_of_years = (data_length - 1) / 365

    return [
        f"Our System Annualized return % => {round(((math.exp(math.log(stats.money/10000.0)/number_of_years)-1)*100), 2)}%",
        f"BaH Annualized return % => {round(((math.exp(math.log(money_bah/10000.0)/number_of_years)-1)*100), 2)}%",
        f"Annualized number of transaction => {round((float(stats.transaction_count)/number_of_years), 1)}#",
        f"Percent success of transaction => {round((float(stats.success_transaction_count)/float(stats.transaction_count))*100, 2)}%",
        f"Average percent profit per transaction => {round((stats.total_percent_profit/stats.transaction_count*100), 2)}%",
        f"Average transaction length => {stats.total_transaction_length//stats.transaction_count}#",
        f"Maximum profit percent in transaction=> {round(stats.maximum_gain / 100, 2)}%",
        f"Maximum loss percent in transaction=> {round(stats.maximum_lost / 100, 2)}%",
        f"Maximum capital value=> ${round(stats.maximum_money, 2)}",
        f"Minimum capital value=> ${round(stats.minimum_money, 2)}",
        f"Idle Ratio %=>  {round((float(data_length-stats.total_transaction_length)/float(data_length)*100), 2)}%"
    ]

def write_to_file(data: List[str], file_path: str) -> None:
    try:
        with open(file_path, "w") as writer:
            writer.writelines(line + '\n' for line in data)
    except IOError as e:
        print(f"An error occurred while writing the file: {e}")

def phase_process() -> None:
    fname = "resources2/outputOfTestPrediction.txt"
    data = read_csv_file(fname)

    stats = process_transactions(data)
    money_bah = calculate_bah(data)

    builder = [
        f"Start Capital: $10000.0",
        f"Our System => totalMoney = ${round(stats.money, 2)}",
        f"BAH => totalMoney = ${round(money_bah, 2)}"
    ]

    results = generate_report(stats, money_bah, len(data))
    builder.extend(results)

    for result in results:
        print(result)

    write_to_file(builder, "resources2/Results.txt")
    print("Results have been written to resources2/Results.txt")

if __name__ == "__main__":
    phase_process()