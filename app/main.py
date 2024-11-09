from app.preprocessing.feature_proprocessor import FeatureMaker
from app.data.datahelper import TestDataGenerator
from app.algorithm.MLPTrader import ModelConfig, MLTrader
from app.backtesting.backtest_system import TradingSystem
from app.algorithm.GA import GA
from app.visualization.backtest_stat import BackTestStatisticVisualizer
from pathlib import Path
import shutil
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PathConfig:
    """Centralized configuration for all file paths"""
    
    # Base paths
    BASE_DIR = Path(os.getenv('BASE_DIR', 'C:/Users/Steve/Desktop/Projects/fyp'))
    DATA_DIR = BASE_DIR / 'app' / 'data' / 'stock_data'
    
    # Input/Output paths
    OUTPUT_CSV = DATA_DIR / 'output.csv'
    OUTPUT_MLP = DATA_DIR / 'outputMLP.csv'
    OUTPUT_TEST_PREDICTION = DATA_DIR / 'outputOfTestPrediction.txt'
    GA_TRAINING_LIST = DATA_DIR / 'GATableListTraining.txt'
    GA_TEST_LIST = DATA_DIR / 'GATableListTest.txt'
    
    @classmethod
    def get_company_dir(cls, company: str) -> Path:
        """Get company-specific directory path"""
        return cls.DATA_DIR / company
    
    @classmethod
    def get_company_training_file(cls, company: str) -> Path:
        """Get company training period file path"""
        return cls.DATA_DIR / f"{company}19972007.csv"
    
    @classmethod
    def get_company_test_file(cls, company: str) -> Path:
        """Get company test period file path"""
        return cls.DATA_DIR / f"{company}20072017.csv"

class Scheduler:
    CREATE_TEST_FILE = 0
    CALCULATE = 1
    
    COMPANIES = [
            'AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS',
            'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
            'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'VZ', 'WMT', 'XOM'
    ]

    CLEANUP_FILES = []
    mode = CREATE_TEST_FILE

    @staticmethod
    def copy_company_files(company):
        """Copy company CSV files and GATableListTraining.txt from inner folder to data/stock_data folder"""
        source_dir = PathConfig.get_company_dir(company)
        
        files_to_copy = [
            f"{company}19972007.csv",
            f"{company}20072017.csv",
            "GATableListTraining.txt"
        ]
        
        for file_name in files_to_copy:
            source_path = source_dir / file_name
            dest_path = PathConfig.DATA_DIR / file_name
            try:
                shutil.copy2(source_path, dest_path)
                print(f"Successfully copied: {file_name}")
            except FileNotFoundError:
                print(f"Warning: {file_name} not found")
                if file_name.endswith('.csv'):
                    return False
        return True

    @staticmethod
    def cleanup_files(company):
        """Clean up all generated and copied files"""
        company_files = [
            PathConfig.get_company_training_file(company),
            PathConfig.get_company_test_file(company)
        ]
        
        for file_path in company_files:
            try:
                file_path.unlink()
                print(f"Removed: {file_path}")
            except FileNotFoundError:
                print(f"File not found: {file_path}")
        
        for file in Scheduler.CLEANUP_FILES:
            file_path = PathConfig.DATA_DIR / file
            try:
                file_path.unlink()
                print(f"Removed: {file_path}")
            except FileNotFoundError:
                print(f"File not found: {file_path}")

    @staticmethod
    def process_company(company):
        print(f"\nProcessing company: {company}")
        
        processor = FeatureMaker()
        test_data_getter = TestDataGenerator(
            str(PathConfig.OUTPUT_CSV),
            str(PathConfig.GA_TEST_LIST)
        )
        
        custom_params = {
            'rsi': {'periods': range(1, 21)},
            'sma': {'periods': [50, 200]}
        }

        print("Phase 0 + 1")
        processor.run_analysis(
            input_file_path=str(PathConfig.get_company_test_file(company)),
            output_file_path=str(PathConfig.OUTPUT_CSV),
            features=['rsi', 'sma'],
            custom_params=custom_params
        )

        print("Phase2")
        test_data_getter.process()

        print("Phase3")
        config = ModelConfig(
            hidden_layers=[20, 10, 8, 6, 5],
            max_iterations=260,
            random_state=1234,
            batch_size="auto"
        )
        
        trader = MLTrader(config)
        (X_train, y_train), (X_test, y_test) = trader.prepare_data(
            str(PathConfig.GA_TRAINING_LIST),
            str(PathConfig.GA_TEST_LIST)
        )
        trader.train(X_train, y_train)
        metrics = trader.evaluate(X_test, y_test)

        print(f"Test set accuracy = {metrics.accuracy}")
        print("\nConfusion matrix:")
        print(metrics.confusion_matrix)
        print("\nClassification Report:")
        print(metrics.classification_report)

        trader.save_results(y_test, X_test, metrics.predictions, str(PathConfig.OUTPUT_MLP))

        print("Phase4")
        trader.process_financial_predictions(
            metrics.predictions,
            str(PathConfig.OUTPUT_CSV),
            str(PathConfig.OUTPUT_TEST_PREDICTION)
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
                    print(f"\nCleaning up files for {company}")
                    Scheduler.cleanup_files(company)
            else:
                print(f"Skipping {company} due to missing files")
            
            print(f"{'='*50}")
            print(f"Completed process for {company}")
            print(f"{'='*50}\n")

            BackTestStatisticVisualizer.create_performance_grid()

if __name__ == "__main__":
    Scheduler.main()