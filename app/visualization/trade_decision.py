import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Callable, Any
from app.utils.path_config import PathConfig
from app.preprocessing.indicators.rsi import RSIIndicator
from app.preprocessing.indicators.sma import SMAIndicator
from app.data.datahelper import DataLoader


class TechnicalIndicatorCalculator:
    """
    Handles calculation of technical indicators for stock market data.
    """
    
    def __init__(self):
        """
        Initializes the calculator with default feature calculators and parameters.
        """
        self.feature_calculators: Dict[str, Callable] = {
            'rsi': self._calculate_rsi_features,
            'sma': self._calculate_sma_features
        }
        
        self.default_params: Dict[str, Dict[str, Any]] = {
            'rsi': {'periods': range(1, 21)},
            'sma': {'periods': [50, 200]}
        }
    
    def calculate_features(self, df: pd.DataFrame, 
                         features: List[str] = None, 
                         custom_params: Dict[str, Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Calculates specified technical indicators for the given data.
        """
        close_prices: np.ndarray = df['close'].values
        
        features = features or list(self.feature_calculators.keys())
        params = {**self.default_params}
        if custom_params:
            params.update(custom_params)
        
        for feature in features:
            if feature in self.feature_calculators:
                df = self.feature_calculators[feature](df, close_prices, params[feature])
        
        return df
    
    def _calculate_rsi_features(self, df: pd.DataFrame, close_prices: np.ndarray, 
                              params: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculates RSI indicators for specified periods with padding.
        """
        def _apply_rsi_padding(df: pd.DataFrame, period: int) -> None:
            df.loc[0, f'rsi{period}'] = 0.0
            df.loc[1, f'rsi{period}'] = 100.0
            
            for j in range(2, len(df)):
                if j < period:
                    df.loc[j, f'rsi{period}'] = df.loc[j, f'rsi{j}']
                    
        for period in params['periods']:
            rsi_indicator: RSIIndicator = RSIIndicator(close_prices, period)
            df[f'rsi{period}'] = [rsi_indicator.calculate(j) for j in range(len(df))]
            _apply_rsi_padding(df, period)
        return df
    
    def _calculate_sma_features(self, df: pd.DataFrame, close_prices: np.ndarray, 
                              params: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculates SMA indicators for specified periods.
        """
        for period in params['periods']:
            sma_indicator: SMAIndicator = SMAIndicator(close_prices, period)
            df[f'sma{period}'] = [sma_indicator.calculate(j) for j in range(len(df))]
        return df

class TradeDecisionVisualizer:
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
        prices = data["price"].values

        # Calculate SMAs
        indicator_calculator = TechnicalIndicatorCalculator()
        sma_data = indicator_calculator.calculate_features(
            pd.DataFrame({'close': prices}),
            features=['sma'],
            custom_params={'sma': {'periods': [50, 200]}}
        )

        # Plot price and SMAs on main chart (ax1)
        ax1.plot(dates, prices, color="gray", alpha=0.6, label="Price")
        ax1.plot(dates, sma_data['sma50'], color="blue", alpha=0.5, label="50-day SMA", linewidth=2)
        ax1.plot(dates, sma_data['sma200'], color="red", alpha=0.5, label="200-day SMA", linewidth=2)

        # Find and analyze trades
        actual_trades = []
        cumulative_returns = np.zeros(len(data))
        k = 0
        while k < len(data) - 1:
            if data.loc[k, "signal"] == 1.0:  # Buy signal
                buy_point = data.loc[k, "price"]
                buy_index = k
                share_number = (10000.0 - 1.0) / buy_point
                force_sell = False

                # Look for sell point
                for j in range(k, len(data) - 1):
                    sell_point = data.loc[j, "price"]
                    money_temp = (share_number * sell_point) - 1.0

                    # Check stop loss
                    if 10000.0 * 0.85 > money_temp:
                        actual_trades.append(
                            (buy_index, j, buy_point, sell_point, True)
                        )
                        # Calculate return for this trade and add to cumulative
                        trade_return = ((sell_point - buy_point) / buy_point) * 100
                        cumulative_returns[j:] += trade_return
                        k = j + 1
                        break

                    # Check sell signal
                    if data.loc[j, "signal"] == 2.0 or force_sell:
                        actual_trades.append(
                            (buy_index, j, buy_point, sell_point, False)
                        )
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
        force_sell_points = [
            (trade[1], trade[3]) for trade in actual_trades if trade[4]
        ]

        # Calculate returns for finding best and worst trades
        trade_returns = [
            ((sell_price - buy_price) / buy_price) * 100
            for _, _, buy_price, sell_price, _ in actual_trades
        ]
        best_trade_idx = trade_returns.index(max(trade_returns)) if trade_returns else -1
        worst_trade_idx = trade_returns.index(min(trade_returns)) if trade_returns else -1

        # Plot trade points on main chart (ax1)
        if buy_points:
            buy_x, buy_y = zip(*buy_points)
            ax1.scatter(buy_x, buy_y, color="green", marker="^", s=500, label="Buy")

        if sell_points:
            sell_x, sell_y = zip(*sell_points)
            ax1.scatter(sell_x, sell_y, color="red", marker="v", s=500, label="Sell")

        if force_sell_points:
            force_x, force_y = zip(*force_sell_points)
            ax1.scatter(
                force_x, force_y, color="black", marker="x", s=500, label="Force Sell"
            )

        # Highlight trades
        for trade_idx, (
            buy_idx,
            sell_idx,
            buy_price,
            sell_price,
            is_force_sell,
        ) in enumerate(actual_trades, 1):
            # Highlight holding period
            ax1.axvspan(buy_idx, sell_idx, color="blue", alpha=0.1)

            # Add special markers for best and worst trades
            if trade_idx - 1 == best_trade_idx:
                ax1.plot(
                    sell_idx,
                    sell_price,
                    marker="*",
                    color="gold",
                    markersize=30,
                    label="Best Trade",
                )
            elif trade_idx - 1 == worst_trade_idx:
                ax1.plot(
                    sell_idx,
                    sell_price,
                    marker="*",
                    color="red",
                    markersize=30,
                    label="Worst Trade",
                )

        # Add summary statistics
        total_trades = len(actual_trades)
        profitable_trades = sum(1 for _, _, buy, sell, _ in actual_trades if sell > buy)
        force_sells = sum(1 for trade in actual_trades if trade[4])
        success_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        best_return = max(trade_returns) if trade_returns else 0
        worst_return = min(trade_returns) if trade_returns else 0
        final_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0

        summary_text = (
            f"Total Trades: {total_trades}\n"
            f"Profitable Trades: {profitable_trades}\n"
            f"Force Sells: {force_sells}\n"
            f"Success Rate: {success_rate:.1f}%\n"
            f"Best Return: {best_return:.1f}%\n"
            f"Worst Return: {worst_return:.1f}%\n"
            f"Final Return: {final_return:.1f}%"
        )
        plt.figtext(
            0.02,
            0.02,
            summary_text,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        # Customize plots
        ax1.set_title(f"Trading Decisions for {company}")
        ax1.set_ylabel("Price")
        ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax1.grid(True, alpha=0.3)

        ax2.grid(True, alpha=0.3)

        # Save plot
        plt.tight_layout()
        save_path = PathConfig.get_trading_plot_path(company)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()