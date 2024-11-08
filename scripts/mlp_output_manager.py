import os
import shutil

def collect_and_rename_files(root_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Walk through all directories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if we're in an outputMLP.csv directory
        if os.path.basename(dirpath) == "outputMLP.csv":
            # Get the company name (ticker) from the path
            # This assumes the ticker is the parent directory of outputMLP.csv
            company_name = os.path.basename(os.path.dirname(dirpath))
            
            for filename in filenames:
                # Check if file starts with "part-00000"
                if filename.startswith("part-00000"):
                    # Create new filename with company name
                    new_filename = f"{company_name}_outputMLP.csv"
                    
                    # Source and destination paths
                    source_file = os.path.join(dirpath, filename)
                    dest_file = os.path.join(output_dir, new_filename)
                    
                    try:
                        # Copy the file to the new location with new name
                        shutil.copy2(source_file, dest_file)
                        print(f"Copied and renamed {source_file} to {dest_file}")
                    except Exception as e:
                        print(f"Error processing {source_file}: {str(e)}")

# Specify the root directory where your company ticker folders are located
# and the output directory where you want to collect all renamed files
root_directory = r"C:\\Users\\Steve\\Desktop\\Projects\\fyp\\resources2"  # Replace with your actual path
output_directory = r"C:\\Users\\Steve\\Desktop\\Projects\\fyp\\MLPoutput"  # Replace with your desired output path

collect_and_rename_files(root_directory, output_directory)