import matplotlib.pyplot as plt
import pandas as pd

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
        plt.savefig(f'C:\\Users\\Steve\\Desktop\\Projects\\fyp\\app\\data\\plots\\trading_decisions\\trading_decisions_{company}.png', bbox_inches='tight', dpi=300)
        plt.close()