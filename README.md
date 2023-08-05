# Disaster-Response-Messages

In this project, I'll apply data engineering skills to analyze data from disaster to build a model for an API that classifies disaster messages.

Apologies for any confusion. Here's an updated version of the README, incorporating all previously provided text.

---

# Disaster Response Message Classifier

In this project, we apply data engineering skills to analyze disaster data from [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster messages.

## Project Overview

This project is designed to analyze and classify disaster response messages. The goal is to classify messages into various categories to help coordinate disaster response efforts.

The project utilizes a multi-output RandomForest classifier to categorize emergency text messages based on the needs communicated in each message. These categories can then be used by appropriate disaster relief agencies to provide aid.

## Project Structure

The project follows a clear directory structure:

DISASTER-RESPONSE-MESSAGES
├── src
│ ├── process_data.py
│ └── train_classifier.py
├── data
│ ├── raw
│ │ ├── messages.csv
│ │ └── categories.csv
│ └── processed
│ └── DisasterResponse.db
├── notebooks
│ ├── ETL Pipeline Preparation.ipynb
│ └── ML Pipeline Preparation.ipynb
├── images
│ ├── category_counts_distribution.png
│ ├── category_message_distribution.png
│ └── message_distribution.png
├── reports
└── models
└── trained_model.pkl

- **src**: Contains the source code of the project.
  - `process_data.py`: This script is responsible for the initial stage of the data pipeline: data loading, cleaning, and saving. It takes in raw message data and category labels, cleans the data, merges it, and then saves it into an SQLite database for further processing.
  - `train_classifier.py`: This script trains the classifier on the cleaned data, tunes the model for better performance using GridSearchCV, evaluates the model's performance, and then saves the model.
- **data**: This directory contains raw and processed data. The raw data consists of 'messages.csv' and 'categories.csv', and the processed data is stored in an SQLite database.

- **notebooks**: Contains Jupyter notebooks used for exploratory data analysis and model experimentation.
  - `ETL Pipeline Preparation.ipynb`
  - `ML Pipeline Preparation.ipynb`
- **images**: Contains images used in the notebooks or reports.
  - `category_counts_distribution.png`
  - `category_message_distribution.png`
  - `message_distribution.png`
- **reports**: Currently empty, but intended to contain files (likely Jupyter notebooks or markdown files) for reporting on the data analysis, model development, and evaluation.

- **models**: Contains saved machine learning models.

  - `trained_model.pkl`

- **app**: Contains files to run the web application.

## Usage

### Data Processing

To run the ETL pipeline that cleans the data and stores it in a SQLite database, navigate to the project's root directory and run the following command in the terminal:

```bash
python3 src/process_data.py data/raw/messages.csv data/raw/categories.csv data/processed/DisasterResponse.db
```

This command will clean the data and save it into a SQLite database named "DisasterResponse.db".

### Training Classifier

After running the ETL pipeline, you can train the classifier by running the following command in the terminal:

```bash
python3 src/train_classifier.py data/processed/DisasterResponse.db models/trained_model.pkl
```

This will train the model, display the model's performance metrics, and save the model as `trained_model.pkl`.

### Running the Web App

Finally, you can start the web app to visualize the data and use the classifier to classify text messages:

1. Run the following command in the terminal to start the server:

```bash
python3 app/run.py
```

2. Open your web browser and go to `http://0.0.0.0:3001/` or whichever port your application is running on.
3. Use the text box to enter a disaster message.
4. Click the `Classify Message` button.
5. The application will display the categories that the message belongs to.

## Contributing

We welcome any contributions to this project. Feel free to fork/clone this repository to make your modifications and improvements. Pull requests are always appreciated.

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.

---
