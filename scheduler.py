from phase1 import FeatureMaker
from phase2 import TestDataGenerator
from phase3 import phase_process as phase3_process
from phase4 import phase_process as phase4_process
from phase5 import phase_process as phase5_process
from GA import GA

class Scheduler:
    CREATE_TEST_FILE = 0
    CALCULATE = 1

    mode = CREATE_TEST_FILE
    input_file_path_phase1 = "resources2/WMT19972007.csv" # Change the path to file here
    input_file_path_phase1_test = "resources2/WMT20072017.csv" # Change the path to file here

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
        phase3_process()

        print("Phase4")
        phase4_process(Scheduler.file_path_output_of_mlp, Scheduler.file_path_output_of_rsi_test)

        print("Phase5")
        phase5_process()

if __name__ == "__main__":
    Scheduler.main()
