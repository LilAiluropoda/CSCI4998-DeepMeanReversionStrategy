from phase1 import FeatureMaker
from phase2 import TestDataGenerator
from phase3 import ModelConfig, MLTrader
from phase5 import TradingSystem
from GA import GA
from pathlib import Path
import shutil
import os
import pandas as pd
from tabulate import tabulate

class ResultsVisualizer:
    """Handles visualization of results in a tabulated grid format."""
    
    @staticmethod
    def create_performance_grid():
        """Creates a tabulated grid view of performance metrics across all companies."""
        try:
            # Read the results CSV
            df = pd.read_csv("data/stock_data/Results.csv")
            
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
            with open('data/stock_data/performance_summary.txt', 'w') as f:
                f.write("Trading Strategy Performance Summary\n")
                f.write("=" * 120 + "\n")  # Increased width to accommodate more columns
                f.write(table)
            
            print("\nPerformance summary has been saved to 'data/stock_data/performance_summary.txt'")
            
        except Exception as e:
            print(f"Error creating performance grid: {str(e)}")

class Scheduler:
    CREATE_TEST_FILE = 0
    CALCULATE = 1
    
    # List of company tickers
    COMPANIES = [
            'AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS',
            'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
            'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'VZ', 'WMT', 'XOM'
    ]

    # List of files to clean up after each run
    CLEANUP_FILES = [
    ]
    
    mode = CREATE_TEST_FILE

    @staticmethod
    def copy_company_files(company):
        """Copy company CSV files and GATableListTraining.txt from inner folder to data/stock_data folder"""
        source_dir = f"data/stock_data/{company}"  # Adjust this to your inner folder path
        dest_dir = "data/stock_data"
        
        # List of files to copy
        files_to_copy = [
            f"{company}19972007.csv",  # training period file
            f"{company}20072017.csv",  # test period file
            "GATableListTraining.txt"  # GA training file
        ]
        
        # Try to copy each file
        for file_name in files_to_copy:
            source_path = os.path.join(source_dir, file_name)
            dest_path = os.path.join(dest_dir, file_name)
            try:
                shutil.copy2(source_path, dest_path)
                print(f"Successfully copied: {file_name}")
            except FileNotFoundError:
                print(f"Warning: {file_name} not found")
                # Only return False if one of the CSV files is missing
                if file_name.endswith('.csv'):
                    return False
                
        return True

    @staticmethod
    def cleanup_files(company):
        """Clean up all generated and copied files"""
        resources_dir = "data/stock_data"
        
        # Clean up company-specific CSV files
        company_files = [
            f"{company}19972007.csv",
            f"{company}20072017.csv"
        ]
        
        # Remove company CSV files
        for file in company_files:
            file_path = os.path.join(resources_dir, file)
            try:
                os.remove(file_path)
                print(f"Removed: {file}")
            except FileNotFoundError:
                print(f"File not found: {file}")
        
        # Remove generated files
        for file in Scheduler.CLEANUP_FILES:
            file_path = os.path.join(resources_dir, file)
            try:
                os.remove(file_path)
                print(f"Removed: {file}")
            except FileNotFoundError:
                print(f"File not found: {file}")

    @staticmethod
    def run_ga():
        print("Running Genetic Algorithm")
        # GA.main()

    @staticmethod
    def process_company(company):
        print(f"\nProcessing company: {company}")
        
        input_file_path_phase1 = f"data/stock_data/{company}19972007.csv"
        input_file_path_phase1_test = f"data/stock_data/{company}20072017.csv"
        
        processor = FeatureMaker()
        test_data_getter = TestDataGenerator("data/stock_data/output.csv", "data/stock_data/GATableListTest.txt")
        
        custom_params = {
            'rsi': {'periods': range(1, 21)},
            'sma': {'periods': [50, 200]}
        }

        print("Phase 0 + 1")
        processor.run_analysis(
            input_file_path=input_file_path_phase1_test,
            output_file_path="data/stock_data/output.csv",
            features=['rsi', 'sma'],
            custom_params=custom_params
        )

        # Scheduler.run_ga()

        print("Phase2")
        test_data_getter.process()

        print("Phase3")
        base_path = Path("data/stock_data")
        train_path = str(base_path / "GATableListTraining.txt")
        test_path = str(base_path / "GATableListTest.txt")
        output_path = str(base_path / "outputMLP.csv")

        config = ModelConfig(
            hidden_layers=[20, 10, 8, 6, 5],
            max_iterations=260,
            random_state=1234,
            batch_size="auto"
        )
        
        trader = MLTrader(config)
        (X_train, y_train), (X_test, y_test) = trader.prepare_data(train_path, test_path)
        trader.train(X_train, y_train)
        metrics = trader.evaluate(X_test, y_test)

        print(f"Test set accuracy = {metrics.accuracy}")
        print("\nConfusion matrix:")
        print(metrics.confusion_matrix)
        print("\nClassification Report:")
        print(metrics.classification_report)

        trader.save_results(y_test, X_test, metrics.predictions, output_path)

        print("Phase4")
        trader.process_financial_predictions(
            metrics.predictions,
            "data/stock_data/output.csv",
            "data/stock_data/outputOfTestPrediction.txt"
        )

        print("Phase5")
        system = TradingSystem(company)
        system.run()

    @staticmethod
    def main():
        for company in Scheduler.COMPANIES:
            print(f"\n{'='*50}")
            print(f"Starting process for {company}")
            print(f"{'='*50}")
            
            if Scheduler.copy_company_files(company):
                try:
                    Scheduler.process_company(company)
                except Exception as e:
                    print(f"Error processing {company}: {str(e)}")
                finally:
                    # Clean up files regardless of success or failure
                    print(f"\nCleaning up files for {company}")
                    Scheduler.cleanup_files(company)
            else:
                print(f"Skipping {company} due to missing files")
            
            print(f"{'='*50}")
            print(f"Completed process for {company}")
            print(f"{'='*50}\n")

            ResultsVisualizer.create_performance_grid()

if __name__ == "__main__":
    Scheduler.main()
