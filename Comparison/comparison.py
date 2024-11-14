import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to read CSV and extract company names and annualized returns
def read_csv_data(file_path):
    # Reading the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Extracting the necessary columns (Company and Annualized_Return)
    company_names = df['Company']
    annualized_returns = df['Annualized_Return']
    
    return company_names, annualized_returns

# Function to plot the comparison as a line chart and save it as a file
def plot_comparison(file1, file2, label1, label2, output_file):
    # Reading data from both CSV files
    company_names_1, annualized_returns_1 = read_csv_data(file1)
    company_names_2, annualized_returns_2 = read_csv_data(file2)
    
    # Ensure both datasets have the same companies in the same order
    if not all(company_names_1 == company_names_2):
        print("Error: Company names do not match between the two datasets.")
        return
    
    # Calculate the difference in performance between PDGA and Basic GA
    performance_diff = np.array(annualized_returns_2) - np.array(annualized_returns_1)
    
    # Setting up the figure
    plt.figure(figsize=(12, 8))
    
    # Plotting the first dataset with blue color for both line and bars
    plt.plot(company_names_1, annualized_returns_1, label=label1, marker='o', color='#1f77b4', linestyle='-', linewidth=2, alpha=0.8) # Blue
    
    # Plotting the second dataset with orange color for both line and bars
    plt.plot(company_names_2, annualized_returns_2, label=label2, marker='^', color='#ff7f0e', linestyle='--', linewidth=2, alpha=0.8) # Orange
    
    # Plotting the difference as a bar chart
    # Blue bar when Basic GA is better, Orange when PDGA is better
    plt.bar(company_names_1, performance_diff, color=['#ff7f0e' if diff > 0 else '#1f77b4' for diff in performance_diff], alpha=0.3)

    # Adding labels and title
    plt.xlabel('Company')
    plt.ylabel('Annualized Return')
    plt.title('Annualized Return Comparison: Basic GA vs PDGA')
    
    # Rotating x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adding a legend
    plt.legend()
    
    # Adding grid lines for better readability
    plt.grid(True)
    
    # Adjust layout for better fit
    plt.tight_layout()
    
    # Calculate summary statistics
    avg_return_ga = np.mean(annualized_returns_1)
    avg_return_pdga = np.mean(annualized_returns_2)
    
    # Add performance summary as text annotation on the chart
    summary_text = (f"Performance Summary:\n"
                    f"Average Annualized Return (Basic GA): {avg_return_ga:.2f}\n"
                    f"Average Annualized Return (PDGA): {avg_return_pdga:.2f}")
    
    # Positioning the text in the upper right corner to avoid blocking points
    plt.text(0.99, 0.99, summary_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='right', 
             bbox=dict(facecolor='white', alpha=0.5))

    # Save the plot as a file (e.g., PNG format)
    plt.savefig(output_file, format='png')
    
    # Optionally, show the plot (comment this out if not needed)
    # plt.show()

    # Print the summary in the console as well
    print("\nPerformance Summary:")
    print(f'Average Annualized Return (Basic GA): {avg_return_ga:.2f}')
    print(f'Average Annualized Return (PDGA): {avg_return_pdga:.2f}')

if __name__ == "__main__":
    # File paths for your two CSV files
    file1 = 'GA_Results.csv'  # Basic GA
    file2 = 'PDGA_Results.csv'  # PDGA
    
    # Labels for the two datasets (you can modify these)
    label1 = 'Basic GA'
    label2 = 'PDGA'
    
    # Output file for the saved chart (e.g., 'comparison_line_chart.png')
    output_file = 'comparison_line_chart.png'
    
    # Plot and save the comparison chart
    plot_comparison(file1, file2, label1, label2, output_file)
    
    print(f"Line chart saved as {output_file}")