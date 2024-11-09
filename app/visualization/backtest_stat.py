import pandas as pd
from tabulate import tabulate

class BackTestStatisticVisualizer:
    """Handles visualization of results in a tabulated grid format."""
    
    @staticmethod
    def create_performance_grid():
        """Creates a tabulated grid view of performance metrics across all companies."""
        try:
            # Read the results CSV
            df = pd.read_csv("C:\\Users\\Steve\\Desktop\\Projects\\fyp\\app\\data\\stock_data\\Results.csv")
            
            # Select and rename columns for display
            display_df = df[[
                'Company',
                'Final_Return',
                'Final_Return_Pct',
                'Annualized_Return',
                'BaH_Return_Pct',
                'BaH_Annualized_Return',
                'Strategy_Outperformance',
                'Annual_Outperformance',
                'Success_Rate',
                'Annual_Transactions',
                'Avg_Profit_per_Trade',
                'Idle_Ratio'
            ]].copy()
            
            # Rename columns for better display
            display_df.columns = [
                'Company',
                'Final($)',
                'Return%',
                'Ann.Ret%',
                'BaH.Ret%',
                'BaH.Ann%',
                'OutPerf%',
                'Ann.OutPerf%',
                'Success%',
                'Ann.Trades',
                'Avg.Profit%',
                'Idle%'
            ]
            
            # Format numeric columns
            numeric_columns = [col for col in display_df.columns if col != 'Company']
            for col in numeric_columns:
                display_df[col] = display_df[col].round(2)
            
            # Sort by Annual Outperformance
            display_df = display_df.sort_values('Ann.OutPerf%', ascending=False)
            
            # Convert DataFrame to list of lists for tabulate
            table_data = [display_df.columns.tolist()] + display_df.values.tolist()
            
            # Create the tabulated view
            table = tabulate(
                table_data[1:],  # Data without header
                headers=table_data[0],  # Header
                tablefmt='grid',
                floatfmt='.2f',
                numalign='right',
                stralign='left'
            )
            
            # Print to console
            print("\nTrading Strategy Performance Summary")
            print("=" * 120)  # Increased width to accommodate more columns
            print(table)
            
            # Save to file
            with open('C:\\Users\\Steve\\Desktop\\Projects\\fyp\\app\\data\\stock_data\\performance_summary.txt', 'w') as f:
                f.write("Trading Strategy Performance Summary\n")
                f.write("=" * 120 + "\n")  # Increased width to accommodate more columns
                f.write(table)
            
            print("\nPerformance summary has been saved to 'data/stock_data/performance_summary.txt'")
            
        except Exception as e:
            print(f"Error creating performance grid: {str(e)}")
