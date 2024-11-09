import csv
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from tabulate import tabulate

@dataclass
class TradingStats:
    money: float = 10000.0
    total_transaction_length: int = 0
    transaction_count: int = 0
    success_transaction_count: int = 0
    failed_transaction_count: int = 0
    maximum_money: float = 0.0
    minimum_money: float = 10000.0
    maximum_gain: float = 0.0
    maximum_lost: float = 100.0
    total_gain: float = 0.0
    total_percent_profit: float = 0.0

class TradingSimulator:
    def __init__(self, data: List[List[str]], chromosome: str):
        self.data = data
        self.genes = list(map(int, chromosome.split()))
        self.stats = TradingStats()

    def simulate(self) -> TradingStats:
        k = 0
        while k < len(self.data) - 1:
            trend = self.get_trend(k)
            if trend > 0:
                k = self.handle_uptrend(k)
            else:
                k = self.handle_downtrend(k)
            k += 1
        return self.stats

    def get_trend(self, k: int) -> float:
        sma50 = float(self.data[k][21])
        sma200 = float(self.data[k][22])
        return sma50 - sma200

    def handle_uptrend(self, k: int) -> int:
        if float(self.data[k][self.genes[5]]) <= float(self.genes[4]):
            return self.execute_trade(k, 0.9, 5, 7, 6)
        return k

    def handle_downtrend(self, k: int) -> int:
        if float(self.data[k][self.genes[1]]) <= float(self.genes[0]):
            return self.execute_trade(k, 0.85, 1, 3, 2)
        return k

    def execute_trade(self, k: int, stop_loss: float, buy_gene: int, sell_gene: int, target_gene: int) -> int:
        buy_point = float(self.data[k][0]) * 100
        share_number = (self.stats.money - 1.0) / buy_point
        force_sell = False

        for j in range(k, len(self.data) - 1):
            sell_point = float(self.data[j][0]) * 100
            money_temp = (share_number * sell_point) - 1.0
            if self.stats.money * stop_loss > money_temp:
                self.stats.money = money_temp
                force_sell = True

            if float(self.data[j][self.genes[sell_gene]]) >= float(self.genes[target_gene]) or force_sell:
                self.update_stats(k, j, buy_point, sell_point, share_number)
                return j + 1
        return k

    def update_stats(self, k: int, j: int, buy_point: float, sell_point: float, share_number: float) -> None:
        gain = sell_point - buy_point
        self.stats.success_transaction_count += 1 if gain > 0 else 0
        self.stats.failed_transaction_count += 1 if gain <= 0 else 0
        self.stats.maximum_gain = max(self.stats.maximum_gain, gain)
        self.stats.maximum_lost = min(self.stats.maximum_lost, gain)
        self.stats.money = (share_number * sell_point) - 1.0
        self.stats.maximum_money = max(self.stats.maximum_money, self.stats.money)
        self.stats.minimum_money = min(self.stats.minimum_money, self.stats.money)
        self.stats.transaction_count += 1
        self.stats.total_percent_profit += gain / buy_point
        self.stats.total_transaction_length += (j - k)
        self.stats.total_gain += gain

def calculate_bah(data: List[List[str]]) -> float:
    money_bah = 10000.0
    buy_point_bah = float(data[0][0])
    share_number_bah = (money_bah - 1.0) / buy_point_bah
    return (float(data[-1][0]) * share_number_bah) - 1.0

def generate_report(stats: TradingStats, money_bah: float, data_length: int, company: str) -> List[List[Any]]:
    number_of_years = (data_length - 1) / 365
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
    headers = [
        "Company", "GA Return ($)", "Rtn % (%)", "Ann.#ofT (Deals)", "Success % (%)", "ApT (%)", "L (Days)", 
        "MpT (%)", "MLT (%)", "MxC ($)", "MinC ($)", "IR (%)"
    ]
    print(tabulate(results, headers=headers, tablefmt="grid"))

def read_csv_file(fname: str) -> List[List[str]]:
    with open(fname, 'r') as file:
        return list(csv.reader(file, delimiter=';'))

def main():
    fname = "resources2/output.csv"
    data = read_csv_file(fname)
    chromosome = "34 5 66 7 39 5 76 7"

    companies = ["WMT"]
    results = []

    for company in companies:
        simulator = TradingSimulator(data, chromosome)
        stats = simulator.simulate()
        money_bah = calculate_bah(data)
        result = generate_report(stats, money_bah, len(data), company)
        results.append(result)

    display_results(results)

if __name__ == "__main__":
    main()
