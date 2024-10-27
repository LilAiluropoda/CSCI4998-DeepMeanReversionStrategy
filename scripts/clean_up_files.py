import os
import shutil


def cleanup_files():
    """
    Delete specified files from resources2 folder.
    """
    # Files to delete
    files_to_delete = [
        "GATableListTest.txt",
        "output.csv",
        "outputMLP.csv",
        "outputOfTestPrediction.txt",
        "Results.txt"
    ]
    
    # Path to resources2 folder
    folder_path = "resources2"
    
    print("Starting cleanup...")
    
    # Delete each file if it exists
    for file_name in files_to_delete:
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_name}")
            else:
                print(f"Not found: {file_name}")
        except Exception as e:
            print(f"Error deleting {file_name}: {str(e)}")
    
    print("Cleanup completed!")

if __name__ == "__main__":
    cleanup_files()
