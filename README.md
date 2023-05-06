# Fact-or-Fiction-Fake-news-prediction.
This repository contains code for performing binary classification on the Liar dataset using Tfidf Vectorizer. It includes pre-processing of data, creation of TfidfVectorizer object, training and evaluation of four models, and hyperparameter tuning on the Random Forest classifier.

Binary Classification with Tfidf Vectorizer
This repository contains code for performing binary classification on the Liar dataset using Tfidf Vectorizer. The Liar dataset consists of statements labeled as either true or false. Our goal is to build machine learning models that can accurately predict the classification of text data.

Requirements
To run the code in this repository, you will need:

Python 3.x
Jupyter Notebook
Pandas
NumPy
Scikit-learn
NLTK
Getting Started
Clone this repository to your local machine.
Install the required packages using pip or conda.
Open the Jupyter Notebook and run the code.
Overview
The code in this repository performs the following tasks:

Preprocesses the data by converting the labels to binary labels and combining the training and validation datasets to form a larger training set.
Creates a TfidfVectorizer object to convert the text data into numerical data that can be used for machine learning models.
Trains and evaluates four models on the transformed training and testing data: logistic regression, Multinomial Naive Bayes, SVM, and Random Forest classifiers.
Performs hyperparameter tuning on the Random Forest classifier using GridSearchCV to improve its performance.
Selects the best performing model based on its accuracy and F1 score.
