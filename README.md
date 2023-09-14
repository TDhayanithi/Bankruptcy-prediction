# Bankruptcy Prediction Project Report

## Problem Statement
The goal of this research is to develop a bankruptcy prediction model based on financial data. This is critical for banks and investors because it allows them to determine if a company is performing well or is in financial danger. In this report, we'll go through everything we did to prepare the data for developing our model.

## Getting to Know the Data
First and foremost, we obtained a large amount of financial data from five ARFF files. These statistics contain a variety of financial information, such as how profitable a firm is, how much cash it has, and a variety of other things. This data set has 65 distinct columns. The most crucial thing we want to know is if a corporation declared bankruptcy ('1') or not ('0').

## Data Cleaning and PreProcessing

### Handling Missing Values
We noticed that a tiny part of the data, about 1.28%, had some missing info. That's not good for our model, so we had to do something about it. We tried two main ways:

1.  **Mean Imputation**: We filled in the missing values by taking the average of the other values in the same column.
2. **MICE Imputation**: (Multiple Imputation by Chained Equations), is used to guess and fill in the missing numbers based on the rest of the data.

After doing this, we didn't have any missing info left in our data.

### Handling Duplicate Rows
Sometimes, there were rows that were exactly the same in our data, which could mess up our model. So, we found 82 of these and deleted them, leaving us with 6,945 rows of unique data.

### Dealing with Outliers
Some numbers in our data were way different from the others. We looked at them using box plots, and it turns out, all 64 of our numerical columns had these unusual numbers. We decided to keep them because they might be important.

### Making the Labels Easy for the Model
To help the computer understand, we changed the labels from '0' and '1' to numeric values 0 and 1.

## Exploratory Data Analysis (EDA)

Before moving on to feature engineering, model selection, and training, it's good to take a look at it. This helps us find out if there are any  patterns or things we should know about. Here's what we did:

#### Feature Distribution Analysis

We can explore the distribution of key financial features. This involves generating summary statistics and visualizations for important attributes

#### Feature Correlation Analysis

We utilised a correlation matrix heatmap to determine whether the numbers in our data are related in any manner and to graphically illustrate the correlation coefficients between different financial parameters.

## Feature Engineering

Now that we understand our data, we can start making it even better so our computer model can predict bankruptcy. Here are some technics we used:

#### Ratio Calculation

- Calculate important financial ratios such as debt-to-equity ratio, current ratio, and return on assets.
- These ratios can provide additional insights into a company's financial health.

#### Time Series Features

- If the dataset contains time-related information, we can create time-based features like moving averages or trends to capture temporal patterns.

## Model Training and Evaluation

### First Try - Original Model

We trained a Random Forest Classifier on the original dataset with the following results:

- Accuracy: 0.9638
- Precision, Recall, and F1-score for Class 0: 0.97, 1.00, 0.98
- Precision, Recall, and F1-score for Class 1: 0.89, 0.31, 0.46

The original model has high accuracy but very low recall for Class 1 (bankruptcy instances), and that means that the model is better at identifying non-bankruptcy instances than bankruptcy instances.

### Handling Imbalanced Data
To balance the class distribution, we used the Synthetic Minority Over-sampling Technique (SMOTE). This rebalance resulted in better recall for bankruptcy cases. This resulted in a greater balance between the two groups, which led in better recall for bankruptcy cases.

### Feature Engineering
We attempted to improve the current data by merging existing indicators such as debt ratios, profit margins, and others. These extra characteristics provided the model with improved prediction skills.

### Dimensionality Reduction with PCA
Principal Component Analysis (PCA) was used to decrease the dimensionality of the dataset. While this resulted in a minor decrease in accuracy, it significantly enhanced the model's memory for Class 1, boosting its capacity to detect bankruptcy cases.

## Model Comparison
We trained many models, including Logistic Regression, Naive Bayes, and SVM, using a variety of data preparation methods, including PCA, Mutual Information (MI) Score-selected features, and Correlation Matrix-selected features. Here are the outcomes:


### Logistic Regression

- PCA-Transformed Data:
  - Precision: 0.92
  - Accuracy: 0.70
  - Recall: 0.70
  - F1-score: 0.78

- MI Score-Selected Features:
  - Precision: 0.93
  - Accuracy: 0.64
  - Recall: 0.64
  - F1-score: 0.74

- Correlation Matrix-Selected Features:
  - Precision: 0.93
  - Accuracy: 0.71
  - Recall: 0.71
  - F1-score: 0.79

### Naive Bayes

- PCA-Transformed Data:
  - Precision: 0.86
  - Accuracy: 0.06
  - Recall: 0.06
  - F1-score: 0.02

- MI Score-Selected Features:
  - Precision: 0.93
  - Accuracy: 0.07
  - Recall: 0.07
  - F1-score: 0.05

- Correlation Matrix-Selected Features:
  - Precision: 0.84
  - Accuracy: 0.06
  - Recall: 0.06
  - F1-score: 0.02

### SVM

- PCA-Transformed Data:
  - Precision: 0.93
  - Accuracy: 0.69
  - Recall: 0.68
  - F1-score: 0.77

- MI Score-Selected Features:
  - Precision: 0.93
  - Accuracy: 0.68
  - Recall: 0.68
  - F1-score: 0.77

- Correlation Matrix-Selected Features:
  - Precision: 0.93
  - Accuracy: 0.72
  - Recall: 0.72
  - F1-score: 0.80

Among these models, the SVM model trained on the original data consistently performed the best, boasting the highest accuracy, F1-score, and weighted average F1-score.

## Deploying the Machine Learning Model on Heroku Cloud Platform.

### Exporting the Trained Model with the Top 10 Features

To enhance the efficiency of our model, we selected the top 10 features (by importance).

### Building the Flask on Heroku.

We created a web application to host our bankruptcy prediction algorithm in order to assure accessibility and usability. Because of its simplicity, speed, scalability, customization, integration, and deployment features, Flask, a Python web framework, serves as the backbone for our online application.


### Streamlined User Experience

Our online software makes predicting if a firm will go bankrupt extremely simple. You may either upload your financial data in Excel format or enter your critical financial information into an easy-to-use form. The software will even fill in some of the fields for you.

After you've entered your data, all you have to do is click the "Predict" button. Our intelligent algorithm will then instantly inform you if the firm is in risk of going bankrupt or is doing OK. It's quite simple and doesn't require you to be a machine learning specialist.


We also create our app   by using flask , then deployed it to Heroku . The files of this part are located into (Flask_deployment) folder. You can access the prediction page of Heroku link: [Heroku](https://bankruptcypred-cf25dd3b7586.herokuapp.com/)


