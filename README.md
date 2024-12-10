# Disaster Response Message Classifier

This project demonstrates the application of data engineering and machine learning to address challenges in disaster response. By leveraging a multi-output RandomForest classifier, the system categorizes emergency messages, enabling better coordination among disaster relief agencies.

## Overview

Disaster response involves processing a high volume of messages to identify and address critical needs efficiently. This project analyzes and classifies messages into predefined categories to support timely and accurate response efforts.

The classifier was developed using a dataset from [Figure Eight](https://appen.com/) (now Appen), comprising real disaster response data. The final deliverable includes a web application where users can input messages for classification into multiple relevant categories.

## Directory Structure

The repository is organized for clarity and ease of use:

```
Disaster-Response-Messages
├── app
│   └── run.py                # Script to launch the web application
├── src
│   ├── process_data.py       # ETL pipeline for data cleaning and database creation
│   └── train_classifier.py   # Script for model training and evaluation
├── data
│   ├── raw
│   │   ├── messages.csv      # Raw messages dataset
│   │   └── categories.csv    # Raw categories dataset
│   └── processed
│       └── DisasterResponse.db # SQLite database with cleaned data
├── notebooks
│   ├── ETL_Pipeline.ipynb    # ETL process exploration and validation
│   └── ML_Pipeline.ipynb     # Model experimentation and tuning
├── images                    # Visualizations for exploratory data analysis
│   ├── category_counts.png
│   ├── message_lengths.png
│   └── top_categories.png
├── models
│   └── trained_model.pkl     # Saved model for inference
└── reports                   # Placeholder for project reports
```

## Key Components

1. **ETL Pipeline**: 
   - Cleans and merges raw datasets.
   - Stores processed data in a SQLite database for reproducibility.

2. **Machine Learning Pipeline**: 
   - Trains a multi-output RandomForest classifier.
   - Includes hyperparameter optimization using GridSearchCV.
   - Evaluates model performance with metrics such as precision, recall, and F1-score.

3. **Web Application**:
   - Provides an interface for users to classify new messages.
   - Displays classification results visually for enhanced usability.

## Getting Started

### Prerequisites

Ensure the following are installed on your system:
- Python 3.8 or higher
- Required Python libraries (install using `requirements.txt`):

```bash
pip install -r requirements.txt
```

### Data Processing

Run the ETL pipeline to prepare the data:

```bash
python3 src/process_data.py data/raw/messages.csv data/raw/categories.csv data/processed/DisasterResponse.db
```

### Model Training

Train and save the classifier:

```bash
python3 src/train_classifier.py data/processed/DisasterResponse.db models/trained_model.pkl
```

### Running the Web App

Launch the web application to classify new messages:

```bash
python3 app/run.py
```

Access the application at `http://127.0.0.1:3001/`.

## Usage Example

1. Enter a disaster-related message (e.g., "We need water and food urgently").
2. Click "Classify Message".
3. View the relevant categories (e.g., `water`, `food`).

## Contributing

We encourage contributions to improve the project. Suggestions, bug fixes, and new features are welcome. Fork the repository, make your changes, and submit a pull request.

### Development Guidelines

- Follow Python best practices for code readability (PEP 8).
- Include tests for new functionality where applicable.
- Document your changes clearly in the README or code comments.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions or issues, feel free to contact us via GitHub.
