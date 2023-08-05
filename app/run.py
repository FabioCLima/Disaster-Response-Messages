"""
run.py
~~~~~~~

This module defines a web application for classifying disaster messages.

The application uses Flask for the web server, Plotly for data visualization,
and SQLAlchemy to interface with the SQLite database.

The application has two endpoints:

- `/` or `/index`: This is the main entry point of the application. It
  prepares visualizations based on the data and returns them.

- `/go`: This endpoint takes a user query as input, predicts the
  classification for the query using a pre-trained model, and returns the
  results.

Module Functions
~~~~~~~~~~~~~~~~

- `check_dir(path: Path)`: Verify the existence of the directory.
- `tokenize_text(text: str) -> List[str]`: Tokenize the text.
- `load_data(database_filepath: Path) -> Tuple[pd.DataFrame, object]`: Load
   data from the SQLite database and a trained model from disk.
- `prepare_visualizations(df: pd.DataFrame) -> List[dict]`: Prepare Plotly
   visualizations based on the input DataFrame.
- `setup_app()`: Set up the application, including loading data and the model.
- `main()`: Run the Flask development server.

Module Variables
~~~~~~~~~~~~~~~~

- `app`: The Flask app object.
- `base_dir`: The base directory of the application.
- `data_dir`: The data directory path.
- `processed_data_path`: The processed data directory path.
- `models_dir`: The models directory path.
- `stopwords_`: A list of English stopwords.
- `lemmatizer`: The WordNet lemmatizer.

In the main section of the script, the Flask development server is run.
"""
# Import standard libraries

import pandas as pd
import json
from typing import List, Tuple
from pathlib import Path
from collections import Counter
from operator import itemgetter


# Import third-party libraries
import re
import numpy as np
import plotly
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from flask import Flask, render_template, request
from plotly.graph_objs import Bar
from joblib import load

nltk_packages = ['punkt', 'wordnet', 'stopwords']
for package in nltk_packages:
    try:
        nltk.data.find(package)
    except LookupError:
        nltk.download(package, quiet=True)

stopwords_ = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)


def check_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Directory {path} does not exist.")


base_dir = Path.cwd()
data_dir = base_dir / 'data'
processed_data_path = data_dir / 'processed'
models_dir = base_dir / 'models'

dirs = [data_dir, processed_data_path, models_dir]

for dir_path in dirs:
    check_dir(dir_path)

app = Flask(__name__)

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

        # Tokenize text
        tokenized = word_tokenize(text)

        # Remove stop words
        filtered_text = [word for word in tokenized if word
                         not in stopwords_]

        # Extract root form of words
        lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word
                            in filtered_text]

        return lemmatized_words

    except Exception as e:
        print(f"Error in tokenization: {e}")
        raise


def load_data(database_filepath: Path) -> Tuple[pd.DataFrame, object]:
    """
    Load data from SQLite database and a trained model from disk.

    Args:
        database_filepath (Path): Path object to the database file.

    Returns:
        Tuple[pd.DataFrame, object]: Tuple of DataFrame loaded from the
        database and trained model.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_query('SELECT * FROM DisasterResponseTable', engine)
    df = df.drop(columns=['id'])

    trained_model_path = models_dir / "trained_model.pkl"
    trained_model = load(trained_model_path)

    return df, trained_model


def prepare_visualizations(df: pd.DataFrame) -> List[dict]:
    """
    Prepare plotly visualizations based on the input dataframe.

    The function creates three plotly bar plots:
    1. Distribution of message genres.
    2. Proportion of messages by category.
    3. Frequency of the top 10 words as a percentage of total.

    Args:
        df (pd.DataFrame): The input dataframe containing disaster messages
                           data.
                           The dataframe is expected to have 'genre' and
                           'message' columns,
                            along with one column per message category.

    Returns:
        figures (List[dict]): A list of dictionaries. Each dictionary contains
                              the data and layout.
                              for a plotly bar plot. These can be directly used
                              for visualization in a flask app.

    Raises:
        ValueError: If the required columns are not found in the input
        dataframe.
    """
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    cat_p = df[df.columns[4:]].sum()/len(df)
    cat_p = cat_p.sort_values(ascending=False)
    cats = list(cat_p.index)

    words_with_repetition = [word for text in df['message'].values for word
                             in tokenize(text)]
    word_count_dict = Counter(words_with_repetition)

    sorted_word_count_dict = dict(sorted(word_count_dict.items(),
                                         key=itemgetter(1),
                                         reverse=True))
    top_10_words = list(sorted_word_count_dict.keys())[:10]
    count_props = 100*np.array(list(sorted_word_count_dict.values())[:10])/df.shape[0]

    figures = [
        # First plot
        {
            'data': [Bar(x=genre_names, y=genre_counts)],
            'layout': {'title': 'Distribution of Message Genres',
                       'yaxis': {'title': "Count"},
                       'xaxis': {'title': "Genre"}}
        },
        # Second plot
        {
            'data': [Bar(x=cats, y=cat_p)],
            'layout': {'title': 'Proportion of Messages by Category',
                       'yaxis': {'title': "Proportion", 'automargin': True},
                       'xaxis': {'title': "Category", 'tickangle': -40,
                                 'automargin': True}}
        },
        # Third plot
        {
            'data': [Bar(x=top_10_words, y=count_props)],
            'layout': {
                        'title': 'Frequency of top 10 words as percentage',
                        'yaxis': {'title': 'Occurrence (Out of 100)',
                                  'automargin': True},
                        'xaxis': {'title': 'Top 10 words', 'automargin': True}}
        }
    ]

    return figures


def setup_app():
    """Set up the application, including loading data and the model.

    Returns:
        df: DataFrame loaded from the database.
        trained_model: Trained model loaded from disk.
    """
    database_filepath = processed_data_path / "DisasterResponse.db"
    df, trained_model = load_data(database_filepath)
    return df, trained_model


# Initialize the app
df, trained_model = setup_app()


@app.route('/')
@app.route('/index')
def index():
    """Main application entry point. Prepare visualizations and return them."""
    figures = prepare_visualizations(df)
    # Convert the plotly objects to JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(figures)]
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('master.html',
                           ids=ids,
                           graphJSON=graphJSON, data_set=df)


@app.route('/go')
def go():
    """Use model to predict classification for query and display results."""
    query = request.args.get('query', '')
    classification_labels = trained_model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    return render_template('go.html',
                           query=query,
                           classification_result=classification_results)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
