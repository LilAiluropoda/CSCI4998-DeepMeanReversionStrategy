import numpy as np
import pandas as pd
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import Unthresholded
from pyrqa.computation import RQAComputation, RPComputation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.seasonal import STL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and prepare the data
def load_data(file_path):
    # Read close prices
    df = pd.read_csv(file_path, header=None)
    close_prices = df[0].apply(lambda x: float(x.split(';')[0])).values
    
    # Calculate returns: log(P_t/P_{t-1}) = log(P_t) - log(P_{t-1})
    returns = np.diff(np.log(close_prices))
    return returns, close_prices

def perform_stl_analysis(close_prices, period=22):  # 22 for monthly seasonality (trading days)
    """
    Perform STL decomposition on close prices
    
    Parameters:
    close_prices : array-like
        The close price time series
    period : int
        The seasonal period (default=22 for monthly trading days)
    
    Returns:
    result : STL object
        Contains trend, seasonal, and residual components
    """
    # Convert to pandas Series (STL requires a Series object)
    prices_series = pd.Series(close_prices)
    
    # Perform STL decomposition
    stl = STL(prices_series, 
              period=period,
              robust=True)
    result = stl.fit()
    
    return result

def plot_stl_decomposition(close_prices, period=22):
    """
    Plot the STL decomposition components
    """
    # Perform decomposition
    result = perform_stl_analysis(close_prices, period)
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot original data
    axes[0].plot(close_prices, label='Original')
    axes[0].set_title('Original Time Series')
    axes[0].set_xlabel('Time Points')
    axes[0].set_ylabel('Price')
    
    # Plot trend
    axes[1].plot(result.trend, label='Trend', color='red')
    axes[1].set_title('Trend Component')
    axes[1].set_xlabel('Time Points')
    axes[1].set_ylabel('Trend')
    
    # Plot seasonal
    axes[2].plot(result.seasonal, label='Seasonal', color='green')
    axes[2].set_title('Seasonal Component')
    axes[2].set_xlabel('Time Points')
    axes[2].set_ylabel('Seasonal')
    
    # Plot residual
    axes[3].plot(result.resid, label='Residual', color='purple')
    axes[3].set_title('Residual Component')
    axes[3].set_xlabel('Time Points')
    axes[3].set_ylabel('Residual')
    
    # Add strength of components
    total_var = np.var(result.seasonal) + np.var(result.trend) + np.var(result.resid)
    seasonal_strength = np.var(result.seasonal) / total_var * 100
    trend_strength = np.var(result.trend) / total_var * 100
    residual_strength = np.var(result.resid) / total_var * 100
    
    fig.text(0.01, 0.02, f'Component Strengths:\nTrend: {trend_strength:.1f}%\n'
             f'Seasonal: {seasonal_strength:.1f}%\nResidual: {residual_strength:.1f}%',
             fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return result

# Create recurrence plot
def create_recurrence_plot(data, embed_dim=11, delay=1, threshold=0.16):
    time_series = TimeSeries(data,
                           embedding_dimension=embed_dim,
                           time_delay=delay)
    
    settings = Settings(time_series,
                       analysis_type=Classic,
                       neighbourhood=Unthresholded(),
                       similarity_measure=EuclideanMetric)
    
    computation = RPComputation.create(settings)
    result = computation.run()
    
    return result.recurrence_matrix

# Perform sliding window RQA
def sliding_window_rqa(data, window_size=260, step_size=22, embed_dim=11, delay=1, threshold=0.16):
    n_points = len(data)
    windows = range(0, n_points - window_size, step_size)
    
    rr_values = []  # Recurrence Rate
    det_values = [] # Determinism
    lam_values = [] # Laminarity
    len_values = [] # Average Diagonal Length
    
    for start in windows:
        window_data = data[start:start + window_size]
        result = perform_rqa(window_data, embed_dim, delay, threshold)
        
        rr_values.append(result.recurrence_rate)
        det_values.append(result.determinism)
        lam_values.append(result.laminarity)
        len_values.append(result.average_diagonal_line)
    
    return np.array(rr_values), np.array(det_values), np.array(lam_values), np.array(len_values), windows

# Perform RQA analysis
def perform_rqa(data, embed_dim=11, delay=1, threshold=0.16):
    time_series = TimeSeries(data,
                           embedding_dimension=embed_dim,
                           time_delay=delay)
    
    settings = Settings(time_series,
                       analysis_type=Classic,
                       neighbourhood=FixedRadius(threshold),
                       similarity_measure=EuclideanMetric,
                       theiler_corrector=1)
    
    computation = RQAComputation.create(settings)
    return computation.run()

def main():
    # Load data and calculate returns
    returns, close_prices = load_data('resources2/output.csv')
    
    # Normalize both series
    normalized_returns = (returns - np.mean(returns)) / np.std(returns)
    normalized_price = (close_prices - np.mean(close_prices)) / np.std(close_prices)

    # Create recurrence plots for both
    rec_matrix_returns = create_recurrence_plot(normalized_returns)
    rec_matrix_price = create_recurrence_plot(normalized_price)
    
    # Perform RQA and sliding window analysis for both
    rqa_result_returns = perform_rqa(normalized_returns)
    rqa_result_price = perform_rqa(normalized_price)
    
    rr_ret, det_ret, lam_ret, avg_len_ret, windows_ret = sliding_window_rqa(normalized_returns)
    rr_price, det_price, lam_price, avg_len_price, windows_price = sliding_window_rqa(normalized_price)

    # Create figure with GridSpec
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(4, 2, figure=fig)

    # Add STL analysis
    # plt.figure(figsize=(15, 12))
    # stl_result = plot_stl_decomposition(close_prices)
    
    # Plot 1: Time series (Returns and Price)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(returns)
    ax1.set_title('Returns Time Series')
    ax1.set_xlabel('Time Points')
    ax1.set_ylabel('Returns')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(close_prices)
    ax2.set_title('Price Time Series')
    ax2.set_xlabel('Time Points')
    ax2.set_ylabel('Price')
    
    # Plot 2: Recurrence Plots
    ax3 = fig.add_subplot(gs[1, 0])
    im1 = ax3.imshow(rec_matrix_returns, cmap='binary', aspect='auto')
    ax3.set_title('Recurrence Plot of Returns')
    ax3.set_xlabel('Time Points')
    ax3.set_ylabel('Time Points')
    plt.colorbar(im1, ax=ax3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    im2 = ax4.imshow(rec_matrix_price, cmap='binary', aspect='auto')
    ax4.set_title('Recurrence Plot of Price')
    ax4.set_xlabel('Time Points')
    ax4.set_ylabel('Time Points')
    plt.colorbar(im2, ax=ax4)
    
    # Plot 3: RR and DET
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(windows_ret, rr_ret, label='RR', color='blue')
    ax5.plot(windows_ret, det_ret, label='DET', color='red')
    ax5.set_title('RR and DET (Returns)')
    ax5.set_xlabel('Window Start')
    ax5.set_ylabel('Value')
    ax5.legend()
    
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(windows_price, rr_price, label='RR', color='blue')
    ax6.plot(windows_price, det_price, label='DET', color='red')
    ax6.set_title('RR and DET (Price)')
    ax6.set_xlabel('Window Start')
    ax6.set_ylabel('Value')
    ax6.legend()
    
    # Plot 4: Print RQA measures
    measures_returns = f"""
    Global RQA Measures (Returns):
    Recurrence Rate: {rqa_result_returns.recurrence_rate:.3f}
    Determinism: {rqa_result_returns.determinism:.3f}
    Laminarity: {rqa_result_returns.laminarity:.3f}
    Average Diagonal Length: {rqa_result_returns.average_diagonal_line:.3f}
    """
    
    measures_price = f"""
    Global RQA Measures (Price):
    Recurrence Rate: {rqa_result_price.recurrence_rate:.3f}
    Determinism: {rqa_result_price.determinism:.3f}
    Laminarity: {rqa_result_price.laminarity:.3f}
    Average Diagonal Length: {rqa_result_price.average_diagonal_line:.3f}
    """
    
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.text(0.1, 0.5, measures_returns, fontsize=10, ha='left', va='center')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.text(0.1, 0.5, measures_price, fontsize=10, ha='left', va='center')
    ax8.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()