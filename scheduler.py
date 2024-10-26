from phase1 import FeatureMaker
from phase2 import TestDataGenerator
from phase3 import ModelConfig, MLTrader
from phase5 import TradingSystem
from GA import GA
from pathlib import Path

class Scheduler:
    CREATE_TEST_FILE = 0
    CALCULATE = 1

    company = "JNJ"
    mode = CREATE_TEST_FILE
    input_file_path_phase1 = f"resources2/{company}19972007.csv" # Change the path to file here
    input_file_path_phase1_test = f"resources2/{company}20072017.csv" # Change the path to file here

    file_path_output_of_rsi_test = "resources2/output.csv"
    file_path_output_of_mlp = "resources2/outputMLP.csv"

    @staticmethod
    def run_ga():
        print("Running Genetic Algorithm")
        # GA.main()

    @staticmethod
    def main():
        processor = FeatureMaker()
        test_data_getter = TestDataGenerator(Scheduler.file_path_output_of_rsi_test, "resources2/GATableListTest.txt")
        
        custom_params = {
            'rsi': {'periods': range(1, 21)},
            'sma': {'periods': [50, 200]}
        }

        print("Phase 0 + 1")

        processor.run_analysis(
            input_file_path=Scheduler.input_file_path_phase1_test,
            output_file_path="resources2/output.csv",
            features=['rsi', 'sma'],
            custom_params=custom_params
        )

        # Scheduler.run_ga()

        print("Phase2")
        test_data_getter.process()

        print("Phase3")
        # Define paths
        base_path = Path("resources2")
        train_path = str(base_path / "GATableListTraining.txt")
        test_path = str(base_path / "GATableListTest.txt")
        output_path = str(base_path / "outputMLP.csv")

        # Configure model
        config = ModelConfig(
            hidden_layers=[20, 10, 8, 6, 5],
            max_iterations=200,
            random_state=1234,
            batch_size=128
        )
        
        # Initialize MLTrader
        trader = MLTrader(config)

        # Prepare data
        (X_train, y_train), (X_test, y_test) = trader.prepare_data(
            train_path, test_path
        )

        # Train model
        trader.train(X_train, y_train)

        # Evaluate model
        metrics = trader.evaluate(X_test, y_test)

        # Display results
        print(f"Test set accuracy = {metrics.accuracy}")
        print("\nConfusion matrix:")
        print(metrics.confusion_matrix)
        print("\nClassification Report:")
        print(metrics.classification_report)

        # Save predictions
        trader.save_results(y_test, X_test, metrics.predictions, output_path)

        print("Phase4")
        trader.process_financial_predictions(metrics.predictions, Scheduler.file_path_output_of_rsi_test, "resources2/outputOfTestPrediction.txt")

        print("Phase5")
        system = TradingSystem(Scheduler.company)
        system.run()

if __name__ == "__main__":
    Scheduler.main()
