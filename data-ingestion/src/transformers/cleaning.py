import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataCleaner with a pandas DataFrame.
        
        :param df: Input DataFrame to be cleaned.
        """
        self.df = df

    def remove_duplicates(self):
        """Remove duplicate rows from the dataset."""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        final_count = len(self.df)
        print(f"Removed {initial_count - final_count} duplicate rows.")
        return self

    def handle_missing_values(self, strategy='mean', columns=None):
        """
        Handle missing values in the dataset based on the specified strategy.
        
        :param strategy: Strategy for handling missing values: 'mean', 'median', or 'drop'.
        :param columns: Columns to apply the strategy on. If None, apply on all columns.
        """
        if columns is None:
            columns = self.df.columns

        for col in columns:
            if strategy == 'mean':
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif strategy == 'median':
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif strategy == 'drop':
                self.df = self.df.dropna(subset=[col])
            print(f"Handled missing values in column {col} using strategy: {strategy}")
        
        return self

    def format_date(self, column: str, date_format='%Y-%m-%d'):
        """
        Format date column to a specific format.
        
        :param column: Name of the column containing dates.
        :param date_format: Desired date format, default is '%Y-%m-%d'.
        """
        self.df[column] = pd.to_datetime(self.df[column], format=date_format, errors='coerce')
        print(f"Formatted dates in column {column} to {date_format} format.")
        return self

    def standardize_text(self, column: str):
        """
        Standardize text in a given column by converting to lowercase and stripping whitespaces.
        
        :param column: Name of the column to be standardized.
        """
        self.df[column] = self.df[column].str.lower().str.strip()
        print(f"Standardized text in column {column}.")
        return self

    def remove_outliers(self, column: str, method='iqr'):
        """
        Remove outliers from the dataset using the Interquartile Range (IQR) method or z-score.
        
        :param column: Column to remove outliers from.
        :param method: Method to use, 'iqr' (default) or 'zscore'.
        """
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            initial_count = len(self.df)
            self.df = self.df[~((self.df[column] < (Q1 - 1.5 * IQR)) | (self.df[column] > (Q3 + 1.5 * IQR)))]
            final_count = len(self.df)
            print(f"Removed {initial_count - final_count} outliers from column {column} using IQR.")
        elif method == 'zscore':
            from scipy import stats
            initial_count = len(self.df)
            self.df = self.df[(np.abs(stats.zscore(self.df[column])) < 3)]
            final_count = len(self.df)
            print(f"Removed {initial_count - final_count} outliers from column {column} using Z-score.")
        
        return self

    def normalize_column(self, column: str, method='minmax'):
        """
        Normalize a column using MinMaxScaler or StandardScaler.
        
        :param column: Column to normalize.
        :param method: Normalization method: 'minmax' (default) or 'standard'.
        """
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()

        self.df[column] = scaler.fit_transform(self.df[[column]])
        print(f"Normalized column {column} using {method} scaling.")
        return self

    def drop_columns(self, columns: list):
        """
        Drop specified columns from the DataFrame.
        
        :param columns: List of column names to be dropped.
        """
        self.df = self.df.drop(columns=columns)
        print(f"Dropped columns: {columns}")
        return self

    def add_log_transformation(self, column: str):
        """
        Apply log transformation to the specified column.
        
        :param column: Column to apply log transformation to.
        """
        self.df[column] = np.log1p(self.df[column])
        print(f"Applied log transformation to column {column}.")
        return self

    def categorize_column(self, column: str, bins: int, labels=None):
        """
        Categorize a continuous column into discrete bins.
        
        :param column: Column to categorize.
        :param bins: Number of bins to create.
        :param labels: Labels for the bins. If None, integers will be used.
        """
        self.df[column] = pd.cut(self.df[column], bins=bins, labels=labels)
        print(f"Categorized column {column} into {bins} bins.")
        return self

    def detect_outliers(self, column: str, threshold=3):
        """
        Detect outliers in the specified column based on the z-score.
        
        :param column: Column to detect outliers in.
        :param threshold: Z-score threshold for outlier detection.
        """
        from scipy import stats
        outliers = np.abs(stats.zscore(self.df[column])) > threshold
        outlier_count = outliers.sum()
        print(f"Detected {outlier_count} outliers in column {column} using z-score threshold of {threshold}.")
        return self.df[outliers]

    def impute_custom(self, column: str, value):
        """
        Impute a specific value for missing data in a column.
        
        :param column: Column to impute values in.
        :param value: Value to fill missing data with.
        """
        self.df[column].fillna(value, inplace=True)
        print(f"Imputed missing values in column {column} with custom value: {value}.")
        return self

    def clean(self):
        """Apply all cleaning steps and return cleaned DataFrame."""
        return self.df


# Utility functions for additional processing
def load_data(filepath: str):
    """
    Load dataset from a CSV file.
    
    :param filepath: Path to the CSV file.
    :return: DataFrame loaded from the CSV.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data from {filepath}")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

def save_data(df: pd.DataFrame, filepath: str):
    """
    Save cleaned DataFrame to a CSV file.
    
    :param df: DataFrame to save.
    :param filepath: Path to the destination CSV file.
    """
    df.to_csv(filepath, index=False)
    print(f"Saved cleaned data to {filepath}")


# Usage
if __name__ == '__main__':
    # Load the dataset
    df = load_data('/raw_data.csv')
    
    if df is not None:
        cleaner = DataCleaner(df)
        cleaned_df = (cleaner
                      .remove_duplicates()
                      .handle_missing_values(strategy='mean')
                      .format_date(column='date')
                      .standardize_text(column='description')
                      .remove_outliers(column='price')
                      .normalize_column(column='price', method='minmax')
                      .add_log_transformation(column='sales')
                      .drop_columns(['unnecessary_column'])
                      .categorize_column(column='age', bins=5)
                      .clean())

        # Save the cleaned data
        save_data(cleaned_df, '/cleaned_data.csv')