import csv
from typing import List, Tuple

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

def convert_to_financial_data(board: List[List[str]], data_test: List[List[str]]) -> List[str]:
    builder: List[str] = []
    counter_zeros, counter_ones, counter_twos = 0, 0, 0
    row_price = 0

    for n in range(len(board)):
        # More robust parsing of predictions
        try:
            pred = float(board[n][2].split('.')[0])
            if pred == 0.0:
                counter_zeros += 1
            elif pred == 1.0:
                counter_ones += 1
            elif pred == 2.0:
                counter_twos += 1
        except (ValueError, IndexError):
            continue

        if (n + 1) % 20 == 0:
            # Modified decision threshold
            total_votes = counter_zeros + counter_ones + counter_twos
            if total_votes > 0:
                zero_ratio = counter_zeros / total_votes
                one_ratio = counter_ones / total_votes
                two_ratio = counter_twos / total_votes
                
                if zero_ratio > 0.5:  # Changed from 14 to ratio
                    builder.append(f"{data_test[row_price][0]};0.0\n")
                elif one_ratio > 0.5:
                    builder.append(f"{data_test[row_price][0]};1.0\n")
                elif two_ratio > 0.5:
                    builder.append(f"{data_test[row_price][0]};2.0\n")
                else:
                    # Use most frequent class if no clear majority
                    max_count = max(counter_zeros, counter_ones, counter_twos)
                    if max_count == counter_zeros:
                        builder.append(f"{data_test[row_price][0]};0.0\n")
                    elif max_count == counter_ones:
                        builder.append(f"{data_test[row_price][0]};1.0\n")
                    else:
                        builder.append(f"{data_test[row_price][0]};2.0\n")

            counter_zeros, counter_ones, counter_twos = 0, 0, 0
            row_price += 1

    return builder

def write_to_file(data: List[str], file_path: str) -> None:
    """
    Write the given data to a file.

    Args:
        data (List[str]): The data to write to the file.
        file_path (str): The path of the file to write to.

    Raises:
        IOError: If there's an error writing to the file.
    """
    try:
        with open(file_path, "w") as writer:
            writer.writelines(data)
    except IOError as e:
        print(f"An error occurred while writing the file: {e}")

# add
__all__ = ['phase_process']

def phase_process(output_of_mlp: str, output_of_rsi_test: str) -> None:
    """
    Process MLP output and RSI test data to generate financial predictions.

    Args:
        output_of_mlp (str): Path to the MLP output CSV file.
        output_of_rsi_test (str): Path to the RSI test CSV file.
    """
    try:
        data: List[List[str]] = read_csv_file(output_of_mlp)
        data_test: List[List[str]] = read_csv_file(output_of_rsi_test)

        financial_data: List[str] = convert_to_financial_data(data, data_test)
        
        write_to_file(financial_data, "resources2/outputOfTestPrediction.txt")
        print("Finished processing and writing financial data.")
    except Exception as e:
        print(f"Error in phase4_process: {str(e)}")

if __name__ == "__main__":
    output_of_mlp: str = "resources2/outputOfMLP.csv"
    output_of_rsi_test: str = "resources2/output.csv"
    phase_process(output_of_mlp, output_of_rsi_test)

