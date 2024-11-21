import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

# Suppress the specific warning
warnings.filterwarnings('ignore', category=InterpolationWarning)

def load_stock_price(stock_name):
    """Load stock price data for a given stock"""
    file_path = f"/research/d2/y22/yxchen2/DeepMlpGA/base_enhanced/app/data/stock_data/{stock_name}/{stock_name}19972017.csv"
    columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"]
    df = pd.read_csv(file_path, names=columns)
    
    # Try parsing dates with flexible format
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=False)
    except:
        # If that fails, try removing problematic rows
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
    
    df = df.sort_values('Date')
    return df["Adj Close"].values

def perform_tests(data):
    """Perform sequential ADF tests following Enders (2012) approach"""
    # Clean data
    data = data[~np.isnan(data) & ~np.isinf(data)]
    
    # Calculate log prices
    log_prices = np.log(data)
    
    # First test: Most complex model with trend and constant (Model 1)
    adf_trend = adfuller(log_prices, regression='ct', autolag='AIC')
    
    # KPSS test remains the same
    try:
        kpss_result = kpss(log_prices, regression='c', nlags='auto')
        kpss_stat = kpss_result[0]
    except:
        kpss_stat = np.nan
    
    # Sequential testing but return only the numerical statistics
    if adf_trend[0] < adf_trend[4]['5%']:  # Reject null at 5% level
        return adf_trend[0], kpss_stat
    else:
        # Test simpler model with only constant (Model 2)
        adf_const = adfuller(log_prices, regression='c', autolag='AIC')
        
        if adf_const[0] < adf_const[4]['5%']:
            return adf_const[0], kpss_stat
        else:
            # Test simplest model (Model 3)
            adf_none = adfuller(log_prices, regression='n', autolag='AIC')
            return adf_none[0], kpss_stat

def analyze_all_stocks():
    stocks = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS',
        'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
        'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'VZ', 'WMT', 'XOM']
    
    results = []
    for stock in stocks:
        try:
            prices = load_stock_price(stock)
            adf_stat, kpss_pval = perform_tests(prices)
            results.append([stock, adf_stat, kpss_pval])
            print(f"Processed {stock}")
        except Exception as e:
            print(f"Error processing {stock}: {str(e)}")
    
    return results

def create_table_image(results):
    # Create figure and axis with adjusted size
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axis('tight')
    ax.axis('off')

    # Split data into two halves
    n = len(results)
    mid = (n + 1) // 2
    
    # Prepare data for table
    table_data = []
    headers = ['Index', 'ADF', 'KPSS', 'Index', 'ADF', 'KPSS']
    
    # Function to add significance stars
    def add_stars(value):
        if abs(value) > 3.43:  # 1% level
            return f"{value:.2f}***"
        elif abs(value) > 2.89:  # 5% level
            return f"{value:.2f}**"
        elif abs(value) > 2.58:  # 10% level
            return f"{value:.2f}*"
        return f"{value:.2f}"
    
    # Combine left and right sides
    for i in range(mid):
        left = results[i]
        right = results[i + mid] if i + mid < n else ['', '', '']
        row = [
            left[0],  # Index left
            add_stars(left[1]),  # ADF left
            f"{left[2]:.4f}",  # KPSS left
            right[0] if i + mid < n else '',  # Index right
            add_stars(right[1]) if i + mid < n else '',  # ADF right
            f"{right[2]:.4f}" if i + mid < n else ''  # KPSS right
        ]
        table_data.append(row)

    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.12]*6)
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Remove all borders first
    for key, cell in table._cells.items():
        cell.set_linewidth(0)

    # Add borders for specific rows
    for i in range(len(table._cells) // 6):  # Iterate through rows
        for j in range(6):  # Iterate through columns
            cell = table._cells.get((i, j), None)
            if cell is not None:
                # Header row
                if i == 0:
                    cell.set_linewidth(1)
                    cell.visible_edges = 'TB'  # Top and Bottom edges
                # First data row
                elif i == 1:
                    cell.set_linewidth(1)
                    cell.visible_edges = 'T'  # Bottom, Left, Right edges
                #     elif j == 5:
                #         cell.visible_edges = 'BR'  # Bottom, Right edges
                #     else:
                #         cell.visible_edges = 'B'  # Bottom edge only
                # JPM row
                elif i == 15:
                    cell.set_linewidth(1)
                    cell.visible_edges = 'B'  # Top and Bottom edges
                # Last row
                # elif i == len(table._cells) // 6 - 1:
                #     cell.set_linewidth(1)
                #     if j == 6:
                #         cell.visible_edges = 'B'  # Bottom edge only

    # Style header row
    for i in range(len(headers)):
        table[0, i].set_text_props(weight='bold')
        table[0, i].set_facecolor('white')
    
    # Modified note about significance
    plt.figtext(0.05, 0.02, "Notes: ***, **, * indicate significance at the 1%, 5%, and 10% levels, respectively.", 
                ha='left', fontsize=9)
    
    # Save the figure with minimal margins
    plt.savefig('adf_test_table.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.05)
    plt.close()

def main():
    results = analyze_all_stocks()
    create_table_image(results)

if __name__ == "__main__":
    main()