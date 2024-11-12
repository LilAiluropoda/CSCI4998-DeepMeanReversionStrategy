from pathlib import Path
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class PathConfig:
    """Centralized path configuration for all file paths"""
    # Base directories
    BASE_DIR: Path = Path(os.getenv('BASE_DIR', 'C:/Users/Steve/Desktop/Projects/fyp'))
    DATA_DIR: Path = BASE_DIR / 'app' / 'data' / 'stock_data'
    PLOTS_DIR: Path = BASE_DIR / 'app' / 'data' / 'plots' / 'trading_decisions'
    
    # Input/Output paths
    OUTPUT_CSV: Path = DATA_DIR / 'output.csv'
    OUTPUT_MLP: Path = DATA_DIR / 'outputMLP.csv'
    OUTPUT_TEST_PREDICTION: Path = DATA_DIR / 'outputOfTestPrediction.txt'
    GA_TRAINING_LIST: Path = DATA_DIR / 'GATableListTraining.txt'
    GA_TEST_LIST: Path = DATA_DIR / 'GATableListTest.txt'
    RESULTS_FILE: Path = DATA_DIR / 'Results.csv'
    SUMMARY_FILE: Path = DATA_DIR / 'performance_summary.txt'

    @classmethod
    def get_company_dir(cls, company: str) -> Path:
        """Get company-specific directory path"""
        return cls.DATA_DIR / company

    @classmethod
    def get_company_data_file(cls, company: str) -> Path:
        """Get company full period (1997-2017) file path"""
        return cls.DATA_DIR / f"{company}19972017.csv"
    
    @classmethod
    def get_company_training_file(cls, company: str) -> Path:
        """Get company training period file path"""
        return cls.DATA_DIR / f"{company}_train.csv"

    @classmethod
    def get_company_test_file(cls, company: str) -> Path:
        """Get company test period file path"""
        return cls.DATA_DIR / f"{company}_test.csv"

    @classmethod
    def get_trading_plot_path(cls, company: str) -> Path:
        """Get path for trading decision plot"""
        return cls.PLOTS_DIR / f"trading_decisions_{company}.png"

    @classmethod
    def get_data_file_path(cls, filename: str) -> Path:
        """Get full path for a data file"""
        return cls.DATA_DIR / filename

    @classmethod
    def ensure_directories_exist(cls) -> None:
        """Ensure all required directories exist"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_all_paths(cls) -> dict[str, Path]:
        """Get dictionary of all static paths"""
        return {
            'BASE_DIR': cls.BASE_DIR,
            'DATA_DIR': cls.DATA_DIR,
            'PLOTS_DIR': cls.PLOTS_DIR,
            'OUTPUT_CSV': cls.OUTPUT_CSV,
            'OUTPUT_MLP': cls.OUTPUT_MLP,
            'OUTPUT_TEST_PREDICTION': cls.OUTPUT_TEST_PREDICTION,
            'GA_TRAINING_LIST': cls.GA_TRAINING_LIST,
            'GA_TEST_LIST': cls.GA_TEST_LIST,
            'RESULTS_FILE': cls.RESULTS_FILE,
            'SUMMARY_FILE': cls.SUMMARY_FILE
        }

# Example usage and testing
if __name__ == "__main__":
    # Ensure directories exist
    PathConfig.ensure_directories_exist()
    
    # Print all paths
    for name, path in PathConfig.get_all_paths().items():
        print(f"{name}: {path}")
    
    # Example company-specific paths
    company = "AAPL"
    print(f"\nPaths for {company}:")
    print(f"Company Dir: {PathConfig.get_company_dir(company)}")
    print(f"Training File: {PathConfig.get_company_training_file(company)}")
    print(f"Test File: {PathConfig.get_company_test_file(company)}")
    print(f"Trading Plot: {PathConfig.get_trading_plot_path(company)}")