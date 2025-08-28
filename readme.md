# SMS Spam Classifier

This is an end-to-end machine learning project that automatically classifies SMS messages as either "Spam" or "Ham" (legitimate).
About The Project

The goal of this project is **to build an accurate and reliable spam filter** using **Natural Language Processing (NLP) techniques** and the **Scikit-learn** library. The project covers the entire ML pipeline, from data cleaning and exploratory analysis to model training and performance evaluation.

A key objective was to build a classifier with high precision for the spam class, ensuring that legitimate messages are not incorrectly flagged as spam (minimizing false positives).
Tech Stack

    Python 3
    Pandas
    NLTK (Natural Language Toolkit)
    Numpy
    Scikit-learn
    Jupyter Notebook

Project Workflow

    Data Loading and EDA: The dataset was loaded, column names were cleaned, and an initial exploratory data analysis was performed. This included checking for class imbalance and analyzing the average length of spam vs. ham messages.

    Text Preprocessing: A text cleaning pipeline was established to:

        Remove punctuation and special characters.

        Convert all text to lowercase.

        Tokenize messages into individual words.

        Remove common English stopwords (e.g., 'the', 'a', 'in').

    Feature Engineering (Vectorization): The cleaned text messages were converted into numerical vectors using the TF-IDF (Term Frequency-Inverse Document Frequency) method. This technique reflects how important a word is to a message within the entire collection of messages.

    Model Training: The dataset was split into training (80%) and testing (20%) sets. A Multinomial Naive Bayes classifier, which is well-suited for text classification tasks, was trained on the vectorized training data.

    Model Evaluation: The trained model's performance was evaluated on the unseen test set. Key metrics, including accuracy, precision, recall, F1-score, and the confusion matrix, were analyzed.

## Key Results

The model achieved the following performance on the test set:

    Overall Accuracy: 97%

    Spam Precision: 1.00

        This is an excellent result, indicating that every message the model flagged as spam was actually spam. There were zero false positives.

    Spam Recall: 0.76

        The model successfully identified 76% of all spam messages in the test set.


    True Negatives (Ham correctly identified): 965

    False Positives (Ham incorrectly flagged as Spam): 0

    False Negatives (Spam missed by the model): 36

    True Positives (Spam correctly identified): 114

# How To Run This Project

## Clone the repository to your local machine:
    
    git clone https://github.com/El1syum/scikit_spam_filter.git
  

### (Recommended) Create and activate a virtual environment:

#### For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

#### For Windows

    python -m venv venv
    venv\Scripts\activate

## Install the required dependencies:

    pip install -r requirements.txt

## Launch Jupyter Notebook or Jupyter Lab:
        
    jupyter lab

Open the Spam_Classifier.ipynb notebook and run the cells sequentially.

# Dataset

This project uses the ["SMS Spam Collection Data Set" from the UCI Machine Learning Repository.](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)