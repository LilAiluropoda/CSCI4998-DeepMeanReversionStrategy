import csv
import math
from typing import List

def round_precision(value: float, decimal_places: int) -> float:
    return round(value, decimal_places)

def read_csv_file(fname: str) -> List[List[str]]:
    with open(fname, 'r') as file:
        csv_reader = csv.reader(file, delimiter=';')
        return list(csv_reader)

def main():
    fname = "resources2/output.csv"
    data = read_csv_file(fname)

    i = 0
    totalTransactionLength = 0
    buyPoint = sellPoint = gain = totalGain = moneyTemp = maximumMoney = maximumGain = totalPercentProfit = 0.0
    money = minimumMoney = 10000.0
    maximumLost = 100.0
    shareNumber = 0.0
    transactionCount = successTransactionCount = failedTransactionCount = 0
    buyPointBAH = shareNumberBAH = moneyBAH = 10000.0
    maximumProfitPercent = maximumLostPercent = 0.0
    forceSell = False

    chromosome = "34 5 66 7 39 5 76 7"
    genes = list(map(int, chromosome.split()))

    for i, gene in enumerate(genes):
        print(f"gene{i}: {gene}")

    print(f"Start Capital: ${money}")

    k = 0
    while k < len(data) - 1:
        sma50 = float(data[k][21])
        sma200 = float(data[k][22])
        trend = sma50 - sma200

        if trend > 0:  # upTrend
            if float(data[k][genes[5]]) <= float(genes[4]):
                buyPoint = float(data[k][0]) * 100
                shareNumber = (money - 1.0) / buyPoint
                forceSell = False

                for j in range(k, len(data) - 1):
                    sellPoint = float(data[j][0]) * 100
                    moneyTemp = (shareNumber * sellPoint) - 1.0
                    if money * 0.9 > moneyTemp:
                        money = moneyTemp
                        forceSell = True

                    if float(data[j][genes[7]]) >= float(genes[6]) or forceSell:
                        sellPoint = float(data[j][0]) * 100
                        gain = sellPoint - buyPoint

                        if gain > 0:
                            successTransactionCount += 1
                        else:
                            failedTransactionCount += 1

                        if gain >= maximumGain:
                            maximumGain = gain
                            maximumProfitPercent = (maximumGain / buyPoint) * 100
                        if gain <= maximumLost:
                            maximumLost = gain
                            maximumLostPercent = (maximumLost / buyPoint) * 100

                        moneyTemp = (shareNumber * sellPoint) - 1.0
                        money = moneyTemp
                        if money > maximumMoney:
                            maximumMoney = money
                        if money < minimumMoney:
                            minimumMoney = money

                        transactionCount += 1
                        print(f"{transactionCount}.({k+1}-{j+1}) => {round_precision(gain*shareNumber, 2)} Capital: ${round_precision(money, 2)}")

                        totalPercentProfit += gain / buyPoint
                        totalTransactionLength += (j - k)
                        k = j + 1
                        totalGain += gain
                        break
        else:  # downTrend
            if float(data[k][genes[1]]) <= float(genes[0]):
                buyPoint = float(data[k][0]) * 100
                shareNumber = (money - 1.0) / buyPoint
                forceSell = False

                for j in range(k, len(data) - 1):
                    sellPoint = float(data[j][0]) * 100
                    moneyTemp = (shareNumber * sellPoint) - 1.0
                    if money * 0.85 > moneyTemp:
                        money = moneyTemp
                        forceSell = True

                    if float(data[j][genes[3]]) >= float(genes[2]) or forceSell:
                        sellPoint = float(data[j][0]) * 100
                        gain = sellPoint - buyPoint

                        if gain > 0:
                            successTransactionCount += 1
                        else:
                            failedTransactionCount += 1

                        if gain >= maximumGain:
                            maximumGain = gain
                            maximumProfitPercent = (maximumGain / buyPoint) * 100
                        if gain <= maximumLost:
                            maximumLost = gain
                            maximumLostPercent = (maximumLost / buyPoint) * 100

                        moneyTemp = (shareNumber * sellPoint) - 1.0
                        money = moneyTemp
                        if money > maximumMoney:
                            maximumMoney = money
                        if money < minimumMoney:
                            minimumMoney = money

                        transactionCount += 1
                        print(f"{transactionCount}.({k+1}-{j+1}) => {round_precision(gain*shareNumber, 2)} Capital: ${round_precision(money, 2)}")

                        totalPercentProfit += gain / buyPoint
                        totalTransactionLength += (j - k)
                        k = j + 1
                        totalGain += gain
                        break

        k += 1

    print(f"Our System => totalMoney = ${round_precision(money, 2)}")

    buyPointBAH = float(data[0][0])
    shareNumberBAH = (moneyBAH - 1.0) / buyPointBAH
    moneyBAH = (float(data[-1][0]) * shareNumberBAH) - 1.0

    print(f"BAH => totalMoney = ${round_precision(moneyBAH, 2)}")

    numberOfDays = float(len(data) - 1)
    numberOfYears = numberOfDays / 365

    print(f"Our System Annualized return % => {round_precision(((math.exp(math.log(money/10000.0)/numberOfYears)-1)*100), 2)}%")
    print(f"BaH Annualized return % => {round_precision(((math.exp(math.log(moneyBAH/10000.0)/numberOfYears)-1)*100), 2)}%")
    print(f"Annualized number of transaction => {round_precision((float(transactionCount)/numberOfYears), 1)}#")
    print(f"Percent success of transaction => {round_precision((float(successTransactionCount)/float(transactionCount))*100, 2)}%")
    print(f"Average percent profit per transaction => {round_precision((totalPercentProfit/transactionCount*100), 2)}%")
    print(f"Average transaction length => {totalTransactionLength/transactionCount}#")
    print(f"Maximum profit percent in transaction=> {round_precision(maximumProfitPercent, 2)}%")
    print(f"Maximum loss percent in transaction=> {round_precision(maximumLostPercent, 2)}%")
    print(f"Maximum capital value=> ${round_precision(maximumMoney, 2)}")
    print(f"Minimum capital value=> ${round_precision(minimumMoney, 2)}")
    print(f"Idle Ratio %=>  {round_precision((float(len(data)-totalTransactionLength)/float(len(data))*100), 2)}%")

if __name__ == "__main__":
    main()
