"""
train_classifier.py

This module contains the necessary components to train a machine learning model
on disaster response messages. The model is trained to classify the messages
into different categories.

The module contains the following functions:

- `load_data()`: loads data from a SQLite database.
- `prepare_data()`: prepares the loaded data for machine learning by splitting
it into features and targets.
- `tokenize()`: tokenizes a given text message into individual words,
normalizes it, removes stop words, and performs lemmatization.
- `build_model()`: builds a machine learning pipeline and uses GridSearchCV to
find the best parameters.
- `evaluate_model()`: evaluates a trained model on a test set and prints out
the model performance (f1 score, precision, and recall).
- `save_model()`: saves a trained model into a pickle file.
- `main()`: the main function that controls the execution of the above
functions to complete the machine learning pipeline.

Usage:
python train_classifier.py <input_database_filepath> <output_model_filepath>

Example:
python train_classifier.py ../data/DisasterResponse.db classifier.pkl

This script takes two command line arguments:
1. The file path of the disaster messages database.
2. The file path of the pickle file to save the model to.
"""

# Import standard libraries
import sys
import pickle
from pathlib import Path
from typing import List, Any, Tuple

# Import third-party libraries
import warnings
import gzip
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline


# Set maximum number of rows to display
pd.set_option('display.max_rows', None)

# Check if nltk packages are downloaded, if not download them
nltk_packages = ['punkt', 'wordnet', 'stopwords']
for package in nltk_packages:
    try:
        nltk.data.find(package)
    except LookupError:
        nltk.download(package, quiet=True)

# Ignore warnings
warnings.filterwarnings('ignore')


def check_dir(path: Path) -> None:
    """Check if a directory exists. If not, raise an error."""
    if not path.is_dir():
        raise FileNotFoundError(f"Directory {path} does not exist.")


# Get the current working directory (where your Python script is running from)
base_dir = Path.cwd()
data_dir = base_dir / 'data'
raw_data_path = data_dir / 'raw'
processed_data_path = data_dir / 'processed'
notebooks_path = base_dir / 'notebooks'
images_dir = base_dir / 'images'
reports_dir = base_dir / 'reports'
models_dir = base_dir / 'models'

# Check if directories exist
dirs = [data_dir, raw_data_path, processed_data_path, notebooks_path,
        images_dir, reports_dir, models_dir]

for dir_path in dirs:
    check_dir(dir_path)


def load_data(database_filepath: str) -> pd.DataFrame:
    """
    Load data from a SQLite database into a pandas DataFrame.

    Args:
    database_filepath: The path to the SQLite database file.

    Returns:
    A pandas DataFrame containing the data loaded from the database.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_query(
                           'SELECT *'
                           'FROM DisasterResponseTable', engine
                         )
    df = df.drop(columns=['id'])  # remove the 'id' column
    return df


def prepare_data(
    df: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:

    """
    Prepare data for machine learning by splitting it into features
    and targets.

    Args:
    df: A pandas DataFrame containing the data.

    Returns:
    X: A numpy array containing the features.
    Y: A DataFrame containing the targets.
    target_names: A numpy array containing the names of the target variables.

    Raises:
    KeyError: If a required key is not found in the DataFrame.
    Exception: If any other error occurs.
    """
    try:
        df = df.dropna()

        # Features (input)
        X = df['message'].values

        # Targets (output)
        Y = df.iloc[:, 4:]

        target_names = Y.columns.values

        return X, Y, target_names

    except KeyError as e:
        print(f"Error: The key {e} is not found in the DataFrame. "
              f"Please ensure that the DataFrame contains the required keys.")
        raise

    except Exception as e:
        print(f"An unexpected error occurred during data preparation: {e}")
        raise


# Initialize stopwords and lemmatizer
stopwords_ = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def tokenize(text: str) -> List[str]:
    """
    This function takes in raw text data, and outputs the processed tokens.

    Processing involves several steps:
    1. Normalizing the text to lower case and removing non-alphanumeric
    characters.
    2. Splitting the text into individual words (tokenization).
    3. Removing common words that do not carry much information (stopwords).
    4. Reducing each word to its base form (lemmatization).

    Args:
    text (str): The string message which needs to be tokenized.

    Returns:
    words (List[str]): A list of the base forms of non-stopwords from the
    input text.

    Raises:
    Exception: If the function is unable to tokenize the text for any reason.
    """
    try:
        # Normalize text
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

        # tokenize text
        words = word_tokenize(text)

        # remove stop words
        words = [word for word in words if word not in stopwords_]

        # extract root form of words
        words = [lemmatizer.lemmatize(word, pos='v') for word in words]

        return words

    except Exception as e:
        print(f"Error in tokenization: {e}")
        raise


def build_model() -> GridSearchCV:
    """
    Build a machine learning pipeline using NLTK and Scikit-Learn's Pipeline.
    This pipeline includes text processing (tokenization, TF-IDF) and a
    classifier.

    Returns:
    A GridSearchCV object trained to the optimal model parameters.
    """

    # Machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
    ])

    # Grid search parameters
    parameters = {
        'clf__estimator__n_estimators': [100, 200]
    }

    # Grid search with cross-validation
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model: GridSearchCV,
                   X_test: np.array,
                   Y_test: pd.DataFrame,
                   category_names: List[str]) -> None:
    """
    Evaluate the performance of a trained machine learning model.

    Args:
    model: The trained model.
    X_test: The features in the test set.
    Y_test: The labels in the test set.
    category_names: List of the category names.

    Returns:
    None
    """
    # Predict labels for test data
    y_pred = model.predict(X_test)

    f1_scores = []
    for ind, cat in enumerate(category_names):
        print('Class - {}'.format(cat))
        print(classification_report(Y_test.values[ind],
                                    y_pred[ind],
                                    zero_division=1))
        f1_scores.append(f1_score(Y_test.values[ind],
                                  y_pred[ind],
                                  zero_division=1))

    print(f'Trained Model\nMinimum f1 score - {min(f1_scores)}'
          f'\nBest f1 score - {max(f1_scores)}'
          f'\nMean f1 score - {round(sum(f1_scores)/len(f1_scores), 3)}')

    print("\nBest Parameters:", model.best_params_)


def save_model(model: Any, model_filepath: str) -> None:
    """
    Saves the trained model as a Pickle file.

    Args:
        model : The trained model to be saved.
        model_filepath (str): The location and name of the saved model.

    Returns:
        None
    """
    with gzip.open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """
    The main function that executes the machine learning pipeline. This
    includes:

    1. Loading the database file.
    2. Preparing the data for modeling.
    3. Building the machine learning model.
    4. Training the model with the provided data.
    5. Evaluating the model on the test data.
    6. Saving the trained model to a pickle file.

    This function expects two command line arguments:
    - The file path of the disaster messages database.
    - The file path of the pickle file to save the model to.

    Example Usage:
    `python3 train_classifier.py ../data/DisasterResponse.db classifier.pkl`

    Raises:
    Exception: If the number of command line arguments provided is not equal
    to 3.
    """

    if len(sys.argv) == 3:
        input_database_filepath, output_model_filepath = sys.argv[1:]

        print("Loading data from the input DATABASE...\n"
              f"DATABASE: {input_database_filepath}")
        df = load_data(input_database_filepath)

        print("Preparing data for modeling...")

        X, Y, target_names = prepare_data(df)
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
                                                            X,
                                                            Y,
                                                            test_size=0.3,
                                                            random_state=42)
        print("Building model ...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, target_names)

        save_model(model, output_model_filepath)
        print("Saving model ...\n"
              f"MODEL: {output_model_filepath}")

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == "__main__":
    main()
