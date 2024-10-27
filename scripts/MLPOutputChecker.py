
import csv


def clean_prediction(value: str) -> float:
    """Clean prediction string and convert to float."""
    return float(value.strip().strip('"'))

def compare_predictions(file1_path: str, file2_path: str) -> None:
    """
    Compare prediction values from two different formatted CSV files.
    Only print mismatches with full row information.
    
    Args:
        file1_path (str): Path to first CSV
        file2_path (str): Path to second CSV
    """
    mismatches = 0
    total_rows = 0
    
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        for i, (line1, line2) in enumerate(zip(f1, f2), 1):
            total_rows += 1
            
            try:
                # Clean and extract last number from each line
                pred1 = clean_prediction(line1.split(';')[-1])
                pred2 = clean_prediction(line2.split(';')[-1])
                
                # Only print if there's a mismatch
                if pred1 != pred2:
                    print(f"\nMismatch at row {i}:")
                    print(f"Original full row: {line1.strip()}")
                    print(f"Python full row: {line2.strip()}")
                    print(f"Original prediction: {pred1}")
                    print(f"Python prediction: {pred2}")
                    print("-" * 70)
                    mismatches += 1
            
            except ValueError as e:
                print(f"\nError processing row {i}:")
                print(f"Line 1: {line1.strip()}")
                print(f"Line 2: {line2.strip()}")
                print(f"Error: {str(e)}")
                print("-" * 70)
    
    # Print summary at the end
    print(f"\nSummary:")
    print(f"Total rows compared: {total_rows}")
    print(f"Total mismatches: {mismatches}")
    if total_rows > 0:
        print(f"Match rate: {((total_rows - mismatches) / total_rows) * 100:.2f}%")
    print("-" * 70)

if __name__ == "__main__":
    origin_path = r"C:\\Users\\Steve\\Desktop\\Projects\\fyp\\resources2\\MSFT\\outputMLP.csv\\part-00000-d211ef87-b9a0-46f2-a699-e852a321b9cf.csv"   # Update with actual path
    new_path = r"C:\\Users\\Steve\\Desktop\\Projects\\fyp\\resources2\\outputMLP.csv"  # Update with actual path
    
    try:
        compare_predictions(origin_path, new_path)
        print("Comparison complete!")
    except Exception as e:
        print(f"Error occurred: {str(e)}")