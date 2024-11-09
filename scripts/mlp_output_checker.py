import csv
import os
import subprocess
from datetime import datetime
from pathlib import Path
import shutil

# Import required modules
from phase1 import FeatureMaker
from phase2 import TestDataGenerator
from phase3 import ModelConfig, MLTrader

class MLPComparisonSystem:
    def __init__(self):
        self.resources_base = r"C:\Users\Steve\Desktop\Projects\fyp\resources2"
        self.scheduler_path = r"C:\Users\Steve\Desktop\Projects\fyp\scheduler.py"
        self.mlp_output_dir = r"C:\Users\Steve\Desktop\Projects\fyp\MLPoutput"
        self.venv_python = r"C:\Users\Steve\Desktop\Projects\fyp\venv\Scripts\python.exe"

        # Create results directory
        self.results_dir = os.path.join(self.resources_base, "comparison_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.results_dir, f"match_rates_{timestamp}.txt")
        with open(self.log_file, 'w') as log:
            log.write(f"MLP Output Comparison Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write("-" * 50 + "\n")

    def clean_prediction(self, value: str) -> float:
        """Clean prediction string and convert to float."""
        return float(value.strip().strip('"'))

    def cleanup_output_file(self, file_path: str) -> None:
        """Remove the generated output file if it exists."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up: {file_path}")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def copy_company_files(self, company):
        """Copy required company files to resources2 directory."""
        source_dir = os.path.join(self.resources_base, company)
        
        files_to_copy = [
            f"{company}19972007.csv",
            f"{company}20072017.csv",
            "GATableListTraining.txt"
        ]
        
        for file_name in files_to_copy:
            source_path = os.path.join(source_dir, file_name)
            dest_path = os.path.join(self.resources_base, file_name)
            try:
                shutil.copy2(source_path, dest_path)
                print(f"Copied: {file_name}")
            except FileNotFoundError:
                print(f"Warning: {file_name} not found")
                if file_name.endswith('.csv'):
                    return False
        return True

    def cleanup_company_files(self, company):
        """Clean up temporary company files."""
        files_to_clean = [
            f"{company}19972007.csv",
            f"{company}20072017.csv",
            "GATableListTraining.txt",
            "output.csv",
            "GATableListTest.txt",
            "outputMLP.csv",
            "outputOfTestPrediction.txt"
        ]
        
        for file_name in files_to_clean:
            file_path = os.path.join(self.resources_base, file_name)
            self.cleanup_output_file(file_path)

    def process_company(self, company):
        """Process single company through phases 1-3."""
        print(f"\nProcessing {company}")
        
        input_file_path_phase1 = os.path.join(self.resources_base, f"{company}19972007.csv")
        input_file_path_phase1_test = os.path.join(self.resources_base, f"{company}20072017.csv")
        
        # Phase 1
        processor = FeatureMaker()
        custom_params = {
            'rsi': {'periods': range(1, 21)},
            'sma': {'periods': [50, 200]}
        }
        
        processor.run_analysis(
            input_file_path=input_file_path_phase1_test,
            output_file_path=os.path.join(self.resources_base, "output.csv"),
            features=['rsi', 'sma'],
            custom_params=custom_params
        )

        # Phase 2
        test_data_getter = TestDataGenerator(
            os.path.join(self.resources_base, "output.csv"),
            os.path.join(self.resources_base, "GATableListTest.txt")
        )
        test_data_getter.process()

        # Phase 3
        base_path = Path(self.resources_base)
        train_path = str(base_path / "GATableListTraining.txt")
        test_path = str(base_path / "GATableListTest.txt")
        output_path = str(base_path / "outputMLP.csv")

        config = ModelConfig(
            hidden_layers=[20, 10, 8, 6, 5],
            max_iterations=200,
            random_state=1234,
            batch_size="auto"
        )
        
        trader = MLTrader(config)
        (X_train, y_train), (X_test, y_test) = trader.prepare_data(train_path, test_path)
        trader.train(X_train, y_train)
        metrics = trader.evaluate(X_test, y_test)
        trader.save_results(y_test, X_test, metrics.predictions, output_path)

    def compare_mlp_outputs(self, company):
        """Compare MLP outputs between original and generated files."""
        original_file = os.path.join(self.mlp_output_dir, f"{company}_outputMLP.csv")
        generated_file = os.path.join(self.resources_base, "outputMLP.csv")
        
        mismatches = 0
        total_rows = 0
        
        try:
            with open(original_file, 'r') as f1, open(generated_file, 'r') as f2:
                for i, (line1, line2) in enumerate(zip(f1, f2), 1):
                    total_rows += 1
                    try:
                        pred1 = self.clean_prediction(line1.split(';')[-1])
                        pred2 = self.clean_prediction(line2.split(';')[-1])
                        if pred1 != pred2:
                            mismatches += 1
                    except ValueError:
                        mismatches += 1
            
            match_rate = ((total_rows - mismatches) / total_rows) * 100 if total_rows > 0 else 0
            print(f"{company}: {match_rate:.2f}%")  # Simplified console output
            
            with open(self.log_file, 'a') as log:
                log.write(f"{company}: {match_rate:.2f}%\n")
                
        except Exception as e:
            print(f"{company}: Error")  # Simplified error output
            with open(self.log_file, 'a') as log:
                log.write(f"{company}: Error - {str(e)}\n")

    def cleanup_output_file(self, file_path: str) -> None:
        """Remove the generated output file if it exists."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

    def copy_company_files(self, company):
        """Copy required company files to resources2 directory."""
        source_dir = os.path.join(self.resources_base, company)
        
        files_to_copy = [
            f"{company}19972007.csv",
            f"{company}20072017.csv",
            "GATableListTraining.txt"
        ]
        
        for file_name in files_to_copy:
            source_path = os.path.join(source_dir, file_name)
            dest_path = os.path.join(self.resources_base, file_name)
            try:
                shutil.copy2(source_path, dest_path)
            except FileNotFoundError:
                if file_name.endswith('.csv'):
                    return False
        return True

    def run_all_companies(self):
        """Run complete process for all companies."""
        companies = [
            'AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS',
            'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
            'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'VZ', 'WMT', 'XOM'
        ]

        print("Starting comparison...")
        
        # Track successful comparisons and their rates
        successful_rates = []

        for company in companies:
            try:
                if self.copy_company_files(company):
                    self.process_company(company)
                    
                    # Compare and get match rate
                    original_file = os.path.join(self.mlp_output_dir, f"{company}_outputMLP.csv")
                    generated_file = os.path.join(self.resources_base, "outputMLP.csv")
                    
                    mismatches = 0
                    total_rows = 0
                    
                    with open(original_file, 'r') as f1, open(generated_file, 'r') as f2:
                        for i, (line1, line2) in enumerate(zip(f1, f2), 1):
                            total_rows += 1
                            try:
                                pred1 = self.clean_prediction(line1.split(';')[-1])
                                pred2 = self.clean_prediction(line2.split(';')[-1])
                                if pred1 != pred2:
                                    mismatches += 1
                            except ValueError:
                                mismatches += 1
                    
                    match_rate = ((total_rows - mismatches) / total_rows) * 100 if total_rows > 0 else 0
                    successful_rates.append(match_rate)
                    print(f"{company}: {match_rate:.2f}%")
                    
                    with open(self.log_file, 'a') as log:
                        log.write(f"{company}: {match_rate:.2f}%\n")
                else:
                    print(f"{company}: Missing files")
                    with open(self.log_file, 'a') as log:
                        log.write(f"{company}: Missing required files\n")
            except Exception as e:
                print(f"{company}: Error")
                with open(self.log_file, 'a') as log:
                    log.write(f"{company}: Processing error - {str(e)}\n")
            finally:
                self.cleanup_company_files(company)

        # Calculate and display average match rate
        if successful_rates:
            average_rate = sum(successful_rates) / len(successful_rates)
            print(f"\nAverage match rate: {average_rate:.2f}%")
            print(f"Successfully processed: {len(successful_rates)}/{len(companies)} companies")
            
            with open(self.log_file, 'a') as log:
                log.write("-" * 50 + "\n")
                log.write(f"Average match rate: {average_rate:.2f}%\n")
                log.write(f"Successfully processed: {len(successful_rates)}/{len(companies)} companies\n")
                log.write(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            print("\nNo successful comparisons to average")
            with open(self.log_file, 'a') as log:
                log.write("-" * 50 + "\n")
                log.write("No successful comparisons to average\n")
                log.write(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    try:
        system = MLPComparisonSystem()
        system.run_all_companies()
    except Exception as e:
        print(f"Critical error: {str(e)}")