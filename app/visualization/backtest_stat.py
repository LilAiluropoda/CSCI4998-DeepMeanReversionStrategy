import pandas as pd
from tabulate import tabulate
from typing import List, Dict
from app.utils.path_config import PathConfig

class PerformanceMetrics:
    """Configuration for performance metrics display"""
    
    DISPLAY_COLUMNS: Dict[str, str] = {
        'Company': 'Company',
        'Final_Return': 'Final($)',
        'Final_Return_Pct': 'Return%',
        'Annualized_Return': 'Ann.Ret%',
        'BaH_Return_Pct': 'BaH.Ret%',
        'BaH_Annualized_Return': 'BaH.Ann%',
        'Strategy_Outperformance': 'OutPerf%',
        'Annual_Outperformance': 'Ann.OutPerf%',
        'Success_Rate': 'Success%',
        'Annual_Transactions': 'Ann.Trades',
        'Avg_Profit_per_Trade': 'Avg.Profit%',
        'Idle_Ratio': 'Idle%'
    }
    
    SORT_COLUMN: str = 'Ann.OutPerf%'

class BackTestStatisticVisualizer:
    """Handles visualization of results in a tabulated grid format."""
    
    @staticmethod
    def _load_and_process_data() -> pd.DataFrame:
        """Load and process results data"""
        df = pd.read_csv(PathConfig.RESULTS_FILE)
        
        # Select and rename columns for display
        display_df = df[list(PerformanceMetrics.DISPLAY_COLUMNS.keys())].copy()
        display_df.columns = list(PerformanceMetrics.DISPLAY_COLUMNS.values())
        
        # Format numeric columns
        numeric_columns = [col for col in display_df.columns if col != 'Company']
        for col in numeric_columns:
            display_df[col] = display_df[col].round(2)
        
        # Sort by specified column
        return display_df.sort_values(PerformanceMetrics.SORT_COLUMN, ascending=False)
    
    @staticmethod
    def _create_table(df: pd.DataFrame) -> str:
        """Create formatted table from DataFrame"""
        table_data = [df.columns.tolist()] + df.values.tolist()
        
        return tabulate(
            table_data[1:],  # Data without header
            headers=table_data[0],  # Header
            tablefmt='grid',
            floatfmt='.2f',
            numalign='right',
            stralign='left'
        )
    
    @staticmethod
    def _save_summary(table: str) -> None:
        """Save performance summary to file"""
        with open(PathConfig.SUMMARY_FILE, 'w') as f:
            f.write("Trading Strategy Performance Summary\n")
            f.write("=" * 120 + "\n")
            f.write(table)
    
    @staticmethod
    def create_performance_grid() -> None:
        """Creates a tabulated grid view of performance metrics across all companies."""
        try:
            # Load and process data
            display_df = BackTestStatisticVisualizer._load_and_process_data()
            
            # Create table
            table = BackTestStatisticVisualizer._create_table(display_df)
            
            # Print to console
            print("\nTrading Strategy Performance Summary")
            print("=" * 120)
            print(table)
            
            # Save to file
            BackTestStatisticVisualizer._save_summary(table)
            
            print(f"\nPerformance summary has been saved to '{PathConfig.SUMMARY_FILE}'")
            
        except Exception as e:
            print(f"Error creating performance grid: {str(e)}")

if __name__ == "__main__":
    BackTestStatisticVisualizer.create_performance_grid()