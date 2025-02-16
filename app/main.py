from app.preprocessing.feature_proprocessor import FeatureMaker
from app.data.datahelper import TestDataGenerator
from app.algorithm.MLPTrader import ModelConfig, MLTrader
from app.backtesting.backtest_system import TradingSystem
from app.algorithm.GA import GA
from app.visualization.backtest_stat import BackTestStatisticVisualizer
from app.data.datahelper import DataLoader, TrainTestSplitter
from pathlib import Path
import shutil
import os
import sys
from dotenv import load_dotenv
from app.utils.path_config import PathConfig

class Scheduler:
    CREATE_TEST_FILE = 0
    CALCULATE = 1
    
    COMPANIES = [
            'JPM'
    ]

    CLEANUP_FILES = [
        'GATableListTraining.txt',  # GA training file
        'output.csv',               # Add other generated files
        'outputMLP.csv',
        'outputOfTestPrediction.txt',
        'GATableListTest.txt'
    ]
    mode = CREATE_TEST_FILE

    @staticmethod
    def copy_company_files(company):
        """Copy company CSV files to data/stock_data folder"""
        source_dir = PathConfig.get_company_dir(company)
        
        files_to_copy = [
            f"{company}19972017.csv"
        ]
        
        for file_name in files_to_copy:
            source_path = source_dir / file_name
            dest_path = PathConfig.DATA_DIR / file_name
            try:
                shutil.copy2(source_path, dest_path)
                print(f"Successfully copied: {file_name}")
            except FileNotFoundError:
                print(f"Warning: {file_name} not found")
                return False
        return True

    @staticmethod
    def cleanup_files(company):
        """Clean up all generated and copied files"""
        files_to_cleanup = [
            PathConfig.get_company_training_file(company),
            PathConfig.get_company_test_file(company),
            *[PathConfig.get_data_file_path(file) for file in Scheduler.CLEANUP_FILES]
        ]
        
        for file_path in files_to_cleanup:
            try:
                if file_path.exists():
                    file_path.unlink()
                    print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

    @staticmethod
    def run_ga(company):
        """Run Genetic Algorithm optimization for training data generation"""
        print("Running Genetic Algorithm to generate training data")
        training_file = PathConfig.get_company_training_file(company)
        if not training_file.exists():
            print(f"Error: Training file not found: {training_file}")
            return False
            
        try:
            GA.main()  # This should generate GATableListTraining.txt
            return PathConfig.GA_TRAINING_LIST.exists()
        except Exception as e:
            print(f"Error during GA execution: {e}")
            return False

    @staticmethod
    def process_company(company):
        print(f"\nProcessing company: {company}")
        
        # Process for each year from 1997 to 2012
        for start_year in range(1997, 2013):
            print(f"\n{'*'*30}")
            print(f"Processing year: {start_year}")
            print(f"{'*'*30}")
            
            # Phase 0 + 1: Feature Processing
            processor = FeatureMaker()
            custom_params = {
                'rsi': {'periods': range(1, 21)},
                'sma': {'periods': [50, 200]}
            }

            # Split data with 4 years training, 1 year testing
            data_splitter = TrainTestSplitter()
            data_splitter.save_split_data(
                company_ticker=company,
                start_year=start_year,
                train_years=4,
                test_years=1
            )

            print("Phase 0 + 1: Processing features")
            processor.run_analysis(
                input_file_path=str(PathConfig.get_company_test_file(company)),
                output_file_path=str(PathConfig.OUTPUT_CSV),
                features=['rsi', 'sma'],
                custom_params=custom_params
            )

            # Phase 2: Generate test data
            print("Phase 2: Generating test data")
            test_data_getter = TestDataGenerator(
                str(PathConfig.OUTPUT_CSV),
                str(PathConfig.GA_TEST_LIST)
            )
            test_data_getter.process()

            # Generate training data using GA
            print("Phase 2.5: Generating training data using GA")
            if not Scheduler.run_ga(company):
                print(f"Failed to generate training data using GA for year {start_year}")
                continue

            # Phase 3: Train and evaluate model
            print("Phase 3: Training and evaluating model")
            config = ModelConfig(
                hidden_layers=[20, 10, 8, 6, 5],
                max_iterations=260,
                random_state=1234,
                batch_size="auto"
            )
            
            trader = MLTrader(config)
            try:
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

                # Phase 4: Process predictions
                print("Phase 4: Processing predictions")
                trader.process_financial_predictions(
                    metrics.predictions,
                    str(PathConfig.OUTPUT_CSV),
                    str(PathConfig.OUTPUT_TEST_PREDICTION)
                )

                # Phase 5: Run trading system
                print("Phase 5: Running trading system")
                system = TradingSystem(company, start_year)  # Assuming TradingSystem can accept year parameter
                system.run()

            except Exception as e:
                print(f"Error processing year {start_year}: {str(e)}")
                continue
            
            print(f"Completed processing for year {start_year}")
            Scheduler.cleanup_files(company)

    @staticmethod
    def main():
        # Ensure all required directories exist
        PathConfig.ensure_directories_exist()
        
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