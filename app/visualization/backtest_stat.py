import pandas as pd
from tabulate import tabulate
from typing import List, Dict
from app.utils.path_config import PathConfig

class PerformanceMetrics:
    """Configuration for performance metrics display"""
    
    DISPLAY_COLUMNS: Dict[str, str] = {
        'Company': 'Company',
        'Year': 'Year',
        'Training_Period': 'Train',
        'Testing_Period': 'Test',
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
    
    SORT_COLUMNS: List[str] = ['Company', 'Year']

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
        numeric_columns = [col for col in display_df.columns 
                         if col not in ['Company', 'Year', 'Train', 'Test']]
        for col in numeric_columns:
            display_df[col] = display_df[col].round(2)
        
        # Sort by company and year
        return display_df.sort_values(PerformanceMetrics.SORT_COLUMNS)
    
    @staticmethod
    def _create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics for each company"""
        numeric_columns = [col for col in df.columns 
                         if col not in ['Company', 'Year', 'Train', 'Test']]
        
        summary_df = df.groupby('Company')[numeric_columns].agg({
            'Final($)': 'mean',
            'Return%': 'mean',
            'Ann.Ret%': 'mean',
            'BaH.Ret%': 'mean',
            'BaH.Ann%': 'mean',
            'OutPerf%': 'mean',
            'Ann.OutPerf%': 'mean',
            'Success%': 'mean',
            'Ann.Trades': 'mean',
            'Avg.Profit%': 'mean',
            'Idle%': 'mean'
        }).round(2)
        
        return summary_df
    
    @staticmethod
    def _create_table(df: pd.DataFrame, title: str = "") -> str:
        """Create formatted table from DataFrame"""
        table_data = [df.columns.tolist()] + df.values.tolist()
        
        table = tabulate(
            table_data[1:],  # Data without header
            headers=table_data[0],  # Header
            tablefmt='grid',
            floatfmt='.2f',
            numalign='right',
            stralign='left'
        )
        
        if title:
            table = f"\n{title}\n{'='*len(title)}\n" + table
            
        return table
    
    @staticmethod
    def _save_summary(detailed_table: str, summary_table: str) -> None:
        """Save performance summary to file"""
        with open(PathConfig.SUMMARY_FILE, 'w') as f:
            f.write("Trading Strategy Performance Summary\n")
            f.write("=" * 120 + "\n\n")
            f.write("Detailed Results by Year\n")
            f.write("-" * 120 + "\n")
            f.write(detailed_table)
            f.write("\n\nSummary Statistics by Company\n")
            f.write("-" * 120 + "\n")
            f.write(summary_table)
    
    @staticmethod
    def create_performance_grid() -> None:
        """Creates a tabulated grid view of performance metrics across all companies and years."""
        try:
            # Load and process data
            display_df = BackTestStatisticVisualizer._load_and_process_data()
            
            # Create summary statistics
            summary_df = BackTestStatisticVisualizer._create_summary_statistics(display_df)
            
            # Create tables
            detailed_table = BackTestStatisticVisualizer._create_table(
                display_df, 
                "Detailed Results by Year"
            )
            summary_table = BackTestStatisticVisualizer._create_table(
                summary_df, 
                "Summary Statistics by Company"
            )
            
            # Print to console
            print("\nTrading Strategy Performance Summary")
            print("=" * 120)
            print(detailed_table)
            print("\n" + summary_table)
            
            # Save to file
            BackTestStatisticVisualizer._save_summary(detailed_table, summary_table)
            
            print(f"\nPerformance summary has been saved to '{PathConfig.SUMMARY_FILE}'")
            
        except Exception as e:
            print(f"Error creating performance grid: {str(e)}")

if __name__ == "__main__":
    BackTestStatisticVisualizer.create_performance_grid()