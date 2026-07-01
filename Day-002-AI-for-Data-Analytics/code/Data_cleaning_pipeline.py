import numpy as np
import pandas as pd


def clean_fraud_data(file_path: str) -> pd.DataFrame:
    """Loads the synthetic fraud dataset, handles duplicates, missing values,

    and filters outliers carefully without losing genuine fraud signals.
    """
    # 1. Load the dataset
    print(f"Loading dataset from '{file_path}'...")
    df = pd.read_csv(file_path)
    initial_rows = df.shape[0]
    print(f"Initial Dataset Shape: {df.shape}")

    # 2. Remove Duplicate Records
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - df.shape[0]
    print(f"Removed {duplicates_removed} duplicate rows.")

    # 3. Handle Missing Values
    missing_summary = df.isnull().sum()
    if missing_summary.sum() > 0:
        print("\nMissing values detected:")
        print(missing_summary[missing_summary > 0])

        # Drop rows missing crucial identifiers if they exist
        critical_ids = ["transaction_id", "user_id"]
        df = df.dropna(subset=[col for col in critical_ids if col in df.columns])

        # Impute numeric columns with their median (more robust than mean)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        print("Missing values handled (median imputation applied).")
    else:
        print("No missing values found.")

    # 4. Handle Outliers (Conditional IQR Method)
    # Fraud transactions often look like outliers. We only strip outliers
    # from the legitimate transactions to avoid cleaning away our target signals.
    if "amount" in df.columns and "is_fraud" in df.columns:
        rows_before_outliers = df.shape[0]

        # Isolate legitimate transactions to calculate baseline boundaries
        legit_tx = df[df["is_fraud"] == 0]

        Q1 = legit_tx["amount"].quantile(0.25)
        Q3 = legit_tx["amount"].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Keep rows that are either fraud OR within normal limits if legitimate
        df = df[(df["is_fraud"] == 1) | ((df["amount"] >= lower_bound) & (df["amount"] <= upper_bound))]

        outliers_removed = rows_before_outliers - df.shape[0]
        print(f"Removed {outliers_removed} outlier records from legitimate transactions.")
        print(f"Valid 'amount' bounds for normal transactions: {lower_bound:.2f} to {upper_bound:.2f}")

    # Final summary
    print(f"\nFinal Cleaned Dataset Shape: {df.shape}")
    print(f"Total rows filtered out: {initial_rows - df.shape[0]}")

    return df


# --- Run Pipeline ---
if __name__ == "__main__":
    input_file = "synthetic_fraud_dataset.csv"
    output_file = "cleaned_fraud_dataset.csv"

    try:
        cleaned_df = clean_fraud_data(input_file)
        cleaned_df.to_csv(output_file, index=False)
        print(f"Success! Cleaned data saved to '{output_file}'")
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found. Please verify the file path.")