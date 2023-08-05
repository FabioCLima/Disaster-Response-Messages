"""
process_data.py
===============
This module is a part of a larger project structure designed to load, clean,
and save disaster response message data for further analysis and modeling.
The project follows a clear directory
structure:

- data: Contains raw and processed data. Raw data includes 'messages.csv' and
 'categories.csv'.
- notebooks: Contains Jupyter notebooks for exploratory data analysis and
model experimentation.
- images: Contains images, if any, used in the notebooks or reports.
- reports: Contains files (likely Jupyter notebooks or markdown files) for
reporting on the data analysis, model development, and evaluation.
- models: Contains saved models.

This script, 'process_data.py', is specifically responsible for preparing the
data for analysis. It includes the following functions:

Functions
---------
create_dir(path: Path) -> None:
    Create a directory if it does not already exist.

load_datasets(raw_data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Load the messages and categories datasets from the specified raw data path.

clean_data(messages: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
    Merge the messages and categories DataFrames and clean the resulting
    DataFrame.

save_data(df: pd.DataFrame, database_name: str) -> None:
    Save the DataFrame to an SQLite database.

main():
    Parse command line arguments and call the necessary functions to load,
    clean, and save the data.

This script takes the file paths of the messages and categories datasets, and
the name of a database to save the cleaned data to as command line arguments.
It then loads the datasets, merges them, cleans the data, and stores it in the
specified SQLite database.

Example:
    Run the following command in the terminal to clean the data and save it to
    a database:

    python3 src/process_data.py data/raw/messages.csv
            data/raw/categories.csv DisasterResponse
"""

# Import necessary libraries

from pathlib import Path  # For handling and interacting with file paths
from sqlalchemy import create_engine  # For creating and interacting with DBs
from typing import Tuple
import pandas as pd  # For data manipulation and analysis
import sys


def create_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


# Get the current working directory (where your Python script is running from)
base_dir = Path.cwd()

# Define the directories where your data is stored and will be processed
# The main directory where all data will be stored
data_dir = base_dir / 'data'
# The sub-directory where raw (unprocessed) data files will be stored
raw_data_path = data_dir / 'raw'
# The sub-directory where processed data files will be stored
processed_data_path = data_dir / 'processed'

# Define the directory where your Jupyter notebooks will be stored
notebooks_path = base_dir / 'notebooks'

# Define the directory where your images will be stored
images_dir = base_dir / 'images'

# Define the directory where your reports will be stored
reports_dir = base_dir / 'reports'

# Define the directory where your trained models will be stored
models_dir = base_dir / 'models'

# Create directories if they do not already exist
dirs = [data_dir, raw_data_path, processed_data_path, notebooks_path,
        images_dir, reports_dir, models_dir]

for dir_path in dirs:
    create_dir(dir_path)


def load_datasets(raw_data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the messages and categories datasets from the specified raw data
    path.

    This function reads the 'messages.csv' and 'categories.csv' files from the
    given directory and returns them as Pandas DataFrames.

    Args:
        raw_data_path (Path): The directory where the raw data files are
        stored.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two Pandas
        DataFrames, the first for messages and the second for categories.

    Example:
        messages, categories = load_datasets(Path('data/raw'))
    """

    # Define the path for the messages dataset and load it into a
    # Pandas DataFrame
    messages_path = raw_data_path / 'messages.csv'
    messages = pd.read_csv(messages_path)

    # Define the path for the categories dataset and load it into a
    # Pandas DataFrame
    categories_path = raw_data_path / 'categories.csv'
    categories = pd.read_csv(categories_path)

    return messages, categories


def clean_data(
               messages: pd.DataFrame,
               categories: pd.DataFrame
               ) -> pd.DataFrame:
    """
    Merges the messages and categories DataFrames and cleans the resulting
    DataFrame.

    This function merges the input DataFrames on the 'id' column, then splits
    the 'categories' column into separate columns for each category. Each
    category column is then converted to a binary format.
    The original 'categories' column is dropped, and the DataFrame with the
    new category columns is returned.

    Args:
        messages (pd.DataFrame): A DataFrame containing message data.
        categories (pd.DataFrame): A DataFrame containing category data.

    Returns:
        pd.DataFrame: A DataFrame resulting from the merge and clean
        operations.

    Raises:
        ValueError: If a category value is found that can't be converted to an
        integer.

    Example:
        try:
            cleaned_df = merge_and_clean_data(messages, categories)
        except ValueError as e:
            print("Error occurred:", e)
    """

    # Merge the messages and categories DataFrames on the 'id' column
    df = messages.merge(categories, on='id')

    # Split the 'categories' column into separate columns for each category
    categories = df.categories.str.split(';', expand=True)

    # Use the first row to extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])

    # Rename the columns of 'categories'
    categories.columns = category_colnames

    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        try:
            # Attempt to convert column from string to numeric
            categories[column] = categories[column].astype(int)
        except ValueError:
            print(f"Error: Non-integer value found in column '{column}'.\
                    Please check the data.")
            # Handle the error appropriately - here we just raise it further
            raise

    # Drop the original categories column from 'df'
    df = df.drop('categories', axis=1)

    # Concatenate the original DataFrame with the new 'categories' DataFrame
    df = pd.concat([df, categories], axis=1)

    # Check for empty categories and drop them
    empty_categories = df.columns[df.isnull().all()]
    df = df.drop(empty_categories, axis=1)

    # Drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df: pd.DataFrame, database_name: str) -> None:
    """
    Saves the DataFrame to an SQLite database.

    This function saves the input DataFrame to a SQLite database with the
    provided database name.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        database_name (str): The name of the database file to which the
        DataFrame will be saved.

    Returns:
        None
    """

    # Define the database path
    database_filepath = processed_data_path / f'{database_name}.db'

    # Create an engine that connects to the SQLite database file
    engine = create_engine(f'sqlite:///{database_filepath}')

    # Write the DataFrame to a table in the SQLite database
    df.to_sql('DisasterResponseTable',
              engine,
              index=False,
              if_exists='replace')


def main():
    if len(sys.argv) == 4:

        path_messages, path_categories, database_name = sys.argv[1:]

        # Ensure the provided paths exist
        if not Path(path_messages).exists():
            print(f'Error: {path_messages} does not exist.\
                    Please check the path.')
            sys.exit(1)

        if not Path(path_categories).exists():
            print(f'Error: {path_categories} does not exist.\
                    Please check the path.')
            sys.exit(1)

        print('Loading data from:\n'
              f'    MESSAGES: {path_messages}\n'
              f'    CATEGORIES: {path_categories}')
        messages, categories = load_datasets(raw_data_path)

        print('Cleaning data...')
        df = clean_data(messages, categories)

        print(f'Saving data to:\n    DATABASE: {database_name}')
        save_data(df, database_name)

        print('Cleaned data saved to database.')

    else:
        print('Error: Incorrect number of arguments provided.\n'
              'Please provide the filepaths of the messages and categories'
              'datasets as the first and second argument respectively and'
              'the name of the database to save the cleaned data '
              'to as the third argument.')
        sys.exit(1)


if __name__ == "__main__":
    main()
