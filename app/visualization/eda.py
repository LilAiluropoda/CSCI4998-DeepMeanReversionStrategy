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
from pathlib import Path
from app.utils.path_config import PathConfig


class RecurrenceAnalyzer:
    """Handles recurrence quantification analysis of financial time series."""

    def __init__(self):
        PathConfig.ensure_directories_exist()

    @staticmethod
    def load_data(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """
        Read and prepare price data.

        Args:
            file_path (Path): Path to the input data file

        Returns:
            tuple: (returns, close_prices)
        """
        df = pd.read_csv(file_path, header=None)
        close_prices = df[0].apply(lambda x: float(x.split(";")[0])).values
        returns = np.diff(np.log(close_prices))
        return returns, close_prices

    @staticmethod
    def perform_stl_analysis(close_prices: np.ndarray, period: int = 22) -> STL:
        """
        Perform STL decomposition on close prices.

        Args:
            close_prices (np.ndarray): Array of close prices
            period (int): Seasonal period in days

        Returns:
            STL: Fitted STL decomposition object
        """
        prices_series = pd.Series(close_prices)
        stl = STL(prices_series, period=period, robust=True)
        return stl.fit()

    def plot_stl_decomposition(self, close_prices: np.ndarray, company: str) -> None:
        """
        Plot STL decomposition components.

        Args:
            close_prices (np.ndarray): Array of close prices
            company (str): Company symbol
        """
        result = self.perform_stl_analysis(close_prices)

        fig, axes = plt.subplots(4, 1, figsize=(15, 12))

        # Plot original data
        axes[0].plot(close_prices, label="Original")
        axes[0].set_title("Original Time Series")
        axes[0].set_xlabel("Time Points")
        axes[0].set_ylabel("Price")

        # Plot trend
        axes[1].plot(result.trend, label="Trend", color="red")
        axes[1].set_title("Trend Component")
        axes[1].set_xlabel("Time Points")
        axes[1].set_ylabel("Trend")

        # Plot seasonal
        axes[2].plot(result.seasonal, label="Seasonal", color="green")
        axes[2].set_title("Seasonal Component")
        axes[2].set_xlabel("Time Points")
        axes[2].set_ylabel("Seasonal")

        # Plot residual
        axes[3].plot(result.resid, label="Residual", color="purple")
        axes[3].set_title("Residual Component")
        axes[3].set_xlabel("Time Points")
        axes[3].set_ylabel("Residual")

        # Add strength of components
        total_var = (
            np.var(result.seasonal) + np.var(result.trend) + np.var(result.resid)
        )
        seasonal_strength = np.var(result.seasonal) / total_var * 100
        trend_strength = np.var(result.trend) / total_var * 100
        residual_strength = np.var(result.resid) / total_var * 100

        fig.text(
            0.01,
            0.02,
            f"Component Strengths:\nTrend: {trend_strength:.1f}%\n"
            f"Seasonal: {seasonal_strength:.1f}%\nResidual: {residual_strength:.1f}%",
            fontsize=10,
        )

        plt.tight_layout()

        # Save plot using PathConfig
        plot_path = PathConfig.PLOTS_DIR / f"stl_decomposition_{company}.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()

    @staticmethod
    def create_recurrence_plot(
        data: np.ndarray, embed_dim: int = 11, delay: int = 1, threshold: float = 0.16
    ) -> np.ndarray:
        """Create recurrence plot matrix."""
        time_series = TimeSeries(data, embedding_dimension=embed_dim, time_delay=delay)

        settings = Settings(
            time_series,
            analysis_type=Classic,
            neighbourhood=Unthresholded(),
            similarity_measure=EuclideanMetric,
        )

        computation = RPComputation.create(settings)
        result = computation.run()
        return result.recurrence_matrix

    @staticmethod
    def sliding_window_rqa(
        data: np.ndarray,
        window_size: int = 260,
        step_size: int = 22,
        embed_dim: int = 11,
        delay: int = 1,
        threshold: float = 0.16,
    ) -> tuple:
        """Perform sliding window RQA analysis."""
        n_points = len(data)
        windows = range(0, n_points - window_size, step_size)

        rr_values = []  # Recurrence Rate
        det_values = []  # Determinism
        lam_values = []  # Laminarity
        len_values = []  # Average Diagonal Length

        for start in windows:
            window_data = data[start : start + window_size]
            result = RecurrenceAnalyzer.perform_rqa(
                window_data, embed_dim, delay, threshold
            )

            rr_values.append(result.recurrence_rate)
            det_values.append(result.determinism)
            lam_values.append(result.laminarity)
            len_values.append(result.average_diagonal_line)

        return (
            np.array(rr_values),
            np.array(det_values),
            np.array(lam_values),
            np.array(len_values),
            windows,
        )

    @staticmethod
    def perform_rqa(
        data: np.ndarray, embed_dim: int = 11, delay: int = 1, threshold: float = 0.16
    ):
        """Perform RQA analysis."""
        time_series = TimeSeries(data, embedding_dimension=embed_dim, time_delay=delay)

        settings = Settings(
            time_series,
            analysis_type=Classic,
            neighbourhood=FixedRadius(threshold),
            similarity_measure=EuclideanMetric,
            theiler_corrector=1,
        )

        computation = RQAComputation.create(settings)
        return computation.run()

    def create_full_visualization(
        self,
        company: str,
        returns: np.ndarray,
        close_prices: np.ndarray,
        rec_matrix_returns: np.ndarray,
        rec_matrix_price: np.ndarray,
        rqa_result_returns,
        rqa_result_price,
        rr_ret: np.ndarray,
        det_ret: np.ndarray,
        windows_ret,
        rr_price: np.ndarray,
        det_price: np.ndarray,
        windows_price,
    ):
        """Create and save comprehensive visualization."""
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(4, 2, figure=fig)

        # Plot 1: Time series
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(returns)
        ax1.set_title("Returns Time Series")
        ax1.set_xlabel("Time Points")
        ax1.set_ylabel("Returns")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(close_prices)
        ax2.set_title("Price Time Series")
        ax2.set_xlabel("Time Points")
        ax2.set_ylabel("Price")

        # Plot 2: Recurrence Plots
        ax3 = fig.add_subplot(gs[1, 0])
        im1 = ax3.imshow(rec_matrix_returns, cmap="binary", aspect="auto")
        ax3.set_title("Recurrence Plot of Returns")
        ax3.set_xlabel("Time Points")
        ax3.set_ylabel("Time Points")
        plt.colorbar(im1, ax=ax3)

        ax4 = fig.add_subplot(gs[1, 1])
        im2 = ax4.imshow(rec_matrix_price, cmap="binary", aspect="auto")
        ax4.set_title("Recurrence Plot of Price")
        ax4.set_xlabel("Time Points")
        ax4.set_ylabel("Time Points")
        plt.colorbar(im2, ax=ax4)

        # Plot 3: RR and DET
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(windows_ret, rr_ret, label="RR", color="blue")
        ax5.plot(windows_ret, det_ret, label="DET", color="red")
        ax5.set_title("RR and DET (Returns)")
        ax5.set_xlabel("Window Start")
        ax5.set_ylabel("Value")
        ax5.legend()

        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(windows_price, rr_price, label="RR", color="blue")
        ax6.plot(windows_price, det_price, label="DET", color="red")
        ax6.set_title("RR and DET (Price)")
        ax6.set_xlabel("Window Start")
        ax6.set_ylabel("Value")
        ax6.legend()

        # Plot 4: RQA measures
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
        ax7.text(0.1, 0.5, measures_returns, fontsize=10, ha="left", va="center")
        ax7.axis("off")

        ax8 = fig.add_subplot(gs[3, 1])
        ax8.text(0.1, 0.5, measures_price, fontsize=10, ha="left", va="center")
        ax8.axis("off")

        plt.tight_layout()

        # Save plot using PathConfig
        plot_path = PathConfig.PLOTS_DIR / f"rqa_analysis_{company}.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()

    def analyze_company(self, company: str) -> None:
        """Perform complete analysis for a single company."""
        try:
            # Load data
            data_file = PathConfig.DATA_DIR / "output.csv"
            returns, close_prices = self.load_data(data_file)

            # Normalize data
            normalized_returns = (returns - np.mean(returns)) / np.std(returns)
            normalized_price = (close_prices - np.mean(close_prices)) / np.std(
                close_prices
            )

            # STL Decomposition
            self.plot_stl_decomposition(close_prices, company)

            # Recurrence Analysis
            rec_matrix_returns = self.create_recurrence_plot(normalized_returns)
            rec_matrix_price = self.create_recurrence_plot(normalized_price)

            rqa_result_returns = self.perform_rqa(normalized_returns)
            rqa_result_price = self.perform_rqa(normalized_price)

            # Sliding Window Analysis
            rr_ret, det_ret, lam_ret, avg_len_ret, windows_ret = (
                self.sliding_window_rqa(normalized_returns)
            )
            rr_price, det_price, lam_price, avg_len_price, windows_price = (
                self.sliding_window_rqa(normalized_price)
            )

            # Create visualizations
            self.create_full_visualization(
                company,
                returns,
                close_prices,
                rec_matrix_returns,
                rec_matrix_price,
                rqa_result_returns,
                rqa_result_price,
                rr_ret,
                det_ret,
                windows_ret,
                rr_price,
                det_price,
                windows_price,
            )

            print(f"Analysis completed for {company}")

        except Exception as e:
            print(f"Error analyzing {company}: {str(e)}")


def main():
    analyzer = RecurrenceAnalyzer()

    # List of companies to analyze
    companies = ["AAPL", "MSFT"]  # Add your company list

    for company in companies:
        analyzer.analyze_company(company)


if __name__ == "__main__":
    main()
