import csv
from typing import List

def reverse_file(input_path: str, output_path: str) -> None:
    """
    Reads a CSV file, reverses the order of rows, and writes to a new CSV file.

    This function reads the input CSV file, reverses the order of its rows,
    cleans each line by removing whitespace and quotes, and writes the result
    to the output CSV file.

    Args:
        input_path (str): The path to the input CSV file.
        output_path (str): The path where the reversed CSV file will be saved.

    Returns:
        None
    """
    try:
        with open(input_path, 'r') as input_file:
            lines: List[str] = input_file.readlines()
        
        with open(output_path, 'w', newline='') as output_file:
            writer = csv.writer(output_file, quoting=csv.QUOTE_NONE, escapechar='\\')
            for line in reversed(lines):
                cleaned_line: str = clean_line(line)
                fields: List[str] = cleaned_line.split(',')
                writer.writerow(fields)
                print(cleaned_line)
    except IOError as e:
        print(f"An error occurred while processing the file: {e}")

def clean_line(line: str) -> str:
    """
    Cleans a line by removing whitespace and quotes.

    Args:
        line (str): The input line to clean.

    Returns:
        str: The cleaned line.
    """
    return line.strip().replace('"', '')

def phase_process(input_file_path: str) -> None:
    """
    Processes the input file and creates a reversed version.

    This function calls reverse_file to create a reversed version of the input file.

    Args:
        input_file_path (str): The path to the input CSV file.

    Returns:
        None
    """
    reverse_file(input_file_path, "resources2/reverseFile.csv")

if __name__ == "__main__":
    # Uncomment the path you want to use
    strpath: str = "resources2/APPL/APPL19972007.csv"
    # strpath = "resources2/CAT20072017.csv"
    
    phase_process(strpath)
