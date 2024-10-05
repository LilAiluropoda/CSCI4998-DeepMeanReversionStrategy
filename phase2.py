import csv
from typing import List, Tuple

def phase_process(input_file_path: str) -> None:
    """
    Process the input CSV file and convert it to test data.

    Args:
        input_file_path (str): Path to the input CSV file.

    Returns:
        None
    """
    data: List[List[str]] = read_csv_file(input_file_path)
    convert_to_test_data(data)

def read_csv_file(file_path: str) -> List[List[str]]:
    """
    Read data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        List[List[str]]: A list of rows, where each row is a list of strings.
    """
    data: List[List[str]] = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=';')
        for row in csv_reader:
            data.append(row)
    return data

def convert_to_test_data(board: List[List[str]]) -> None:
    """
    Convert the input data to test data and write it to a file.

    Args:
        board (List[List[str]]): The input data as a list of rows.

    Returns:
        None
    """
    builder: List[str] = []
    for n in range(len(board)):
        sma50, sma200 = get_sma_values(board[n])
        trend_string: str = get_trend_string(sma50, sma200)
        
        for j in range(1, len(board[0]) - 2):  # for each column - except first and last 2 columns (smas)
            builder.append(f"5 1:{board[n][j]} 2:{j} 3:{trend_string}\n")
    
    write_to_file(builder)

def get_sma_values(row: List[str]) -> Tuple[float, float]:
    """
    Extract SMA50 and SMA200 values from a row.

    Args:
        row (List[str]): A row of data.

    Returns:
        Tuple[float, float]: SMA50 and SMA200 values.
    """
    return float(row[21]), float(row[22])

def get_trend_string(sma50: float, sma200: float) -> str:
    """
    Determine the trend based on SMA50 and SMA200 values.

    Args:
        sma50 (float): SMA50 value.
        sma200 (float): SMA200 value.

    Returns:
        str: "1.0" for uptrend, "0.0" for downtrend.
    """
    return "1.0" if sma50 - sma200 > 0 else "0.0"

def write_to_file(data: List[str]) -> None:
    """
    Write the given data to a file.

    Args:
        data (List[str]): The data to write to the file.

    Returns:
        None
    """
    try:
        with open("resources2/GATableListTest.txt", "w") as writer:
            writer.writelines(data)
    except IOError as e:
        print(f"An error occurred while writing the file: {e}")

if __name__ == "__main__":
    input_file_path: str = "resources2/output.csv"
    phase_process(input_file_path)
