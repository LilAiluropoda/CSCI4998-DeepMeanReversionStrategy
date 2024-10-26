import csv
from typing import List, Tuple
from pathlib import Path

class TestDataGenerator:
    """
    A class to process CSV data and convert it into a format suitable for deep learning model testing.
    
    This processor handles:
    - Reading input CSV files
    - Converting financial data into testing format
    - Generating trend indicators
    - Writing processed data to output files
    """
    
    def __init__(self, input_path: str, output_path: str):
        """
        Initialize the data processor with input and output paths.
        
        Args:
            input_path (str): Path to the input CSV file
            output_path (str): Path where the processed data will be written
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.data: List[List[str]] = []
        
    def process(self) -> None:
        """
        Execute the complete data processing workflow.
        
        This method orchestrates the entire process:
        1. Reading the input file
        2. Converting data to testing format
        3. Writing the processed data
        """
        self._read_input_file()
        testing_data = self._convert_to_testing_format()
        self._write_output_file(testing_data)
        
    def _read_input_file(self) -> None:
        """
        Read and store data from the input CSV file.
        
        The CSV file is expected to have semicolon-separated values with
        technical indicators and SMA values.
        """
        try:
            with open(self.input_path, 'r') as file:
                csv_reader = csv.reader(file, delimiter=';')
                self.data = [row for row in csv_reader]
        except IOError as e:
            print(f"Error reading input file: {e}")
            self.data = []
            
    def _convert_to_testing_format(self) -> List[str]:
        """
        Convert the loaded data into the required testing format.
        
        Returns:
            List[str]: Formatted strings ready for testing data file
        
        Format:
            Each line: "5 1:{indicator_value} 2:{column_index} 3:{trend}"
        """
        formatted_data: List[str] = []
        
        for row_index in range(len(self.data)):
            trend = self._get_trend_from_sma(self.data[row_index])
            
            # Process each indicator column (excluding price and SMAs)
            for col_index in range(1, len(self.data[0]) - 2):
                formatted_line = self._format_testing_line(
                    indicator_value=self.data[row_index][col_index],
                    column_index=col_index,
                    trend=trend
                )
                formatted_data.append(formatted_line)
                
        return formatted_data
    
    def _get_trend_from_sma(self, row: List[str]) -> str:
        """
        Calculate trend based on SMA50 and SMA200 values from a data row.
        
        Args:
            row (List[str]): A row of data containing SMA values
            
        Returns:
            str: "1.0" for uptrend (SMA50 > SMA200), "0.0" for downtrend
        """
        sma50 = float(row[21])
        sma200 = float(row[22])
        return "1.0" if sma50 - sma200 > 0 else "0.0"
    
    def _format_testing_line(self, indicator_value: str, column_index: int, trend: str) -> str:
        """
        Format a single line of testing data.
        
        Args:
            indicator_value (str): The technical indicator value
            column_index (int): The index of the indicator column
            trend (str): The calculated trend value
            
        Returns:
            str: Formatted line for testing data file
        """
        return f"5 1:{indicator_value} 2:{column_index} 3:{trend}\n"
    
    def _write_output_file(self, data: List[str]) -> None:
        """
        Write the processed data to the output file.
        
        Args:
            data (List[str]): The formatted testing data to write
        """
        try:
            with open(self.output_path, "w") as writer:
                writer.writelines(data)
        except IOError as e:
            print(f"Error writing output file: {e}")