import pandas as pd
import numpy as np
import os
import argparse


def generate_payment_data(seed, num_samples, fraud_rate, file_path):
    # Create the directory if it does not exist
    output_directory = os.path.dirname(file_path)
    os.makedirs(output_directory, exist_ok=True)

    # Seed for reproducibility
    np.random.seed(seed)

    # Generate random transaction amounts between $10 and $500
    transaction_amounts = np.random.uniform(low=10.0, high=500.0, size=num_samples)

    # Randomly assign transaction types
    transaction_types = np.random.choice(['Online', 'InStore', 'Mobile'], size=num_samples)

    # Randomly assign customer types
    customer_types = np.random.choice(['New', 'Regular', 'VIP'], size=num_samples)

    # Generate fraudulent labels based on the fraud rate
    is_fraudulent = np.random.choice([0, 1], size=num_samples, p=[1 - fraud_rate, fraud_rate])

    # Create a DataFrame
    data = pd.DataFrame({
        'transaction_amount': transaction_amounts,
        'transaction_type': transaction_types,
        'customer_type': customer_types,
        'is_fraudulent': is_fraudulent
    })

    # Save to CSV
    data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate Mock Payment Data")
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generation')
    parser.add_argument('--samples', type=int, default=1000, help='Number of data samples to generate')
    parser.add_argument('--fraud_rate', type=float, default=0.5, help='Fraction of fraudulent transactions')
    parser.add_argument('--file_path', type=str, required=True, help='File path to save the CSV data')

    args = parser.parse_args()

    # Generate the data
    generate_payment_data(args.seed, args.samples, args.fraud_rate, args.file_path)
