# Kaggle-Insurance_Machine_Learning
This machine learning project aims to predict insurance premiums based on various factors. I'll be documenting my machine learning process in this repository, similar to the Kaggle-Depressed_Machine_Learning repository. The latter competition took up a lot of time and i got rewarded with only dark eye bags, and im here to do it all over again. I need to get a life lol.

Once again, feel free to skip to any chapters or versions that interest you, its a long report and its perfectly understandable if you want to just skim certain portions.

## Contents
Chapter 1: Data Cleaning (Versions 1-
- Version 1: A new start
- Version 2: Identifying NaN columns
- Version 3: Tackling NaNs in the 'Age' column

# Chapter 1: Data Cleaning

## Version 1:
Brand new Kaggle notebook, haven't edited anything yet, this is just a standard notebook you get when you create a new notebook in Kaggle. It contains 1 single cell which imports the necessary training and test datasets (train.csv and test.csv), as well as some default packages necessary for machine learning such as numpy and pandas.

## Version 2:
I start with removing the id column (which is unique and doesnt provide insightful patterns or learning points) as well as the target column 'Premium Amount', which i put to one side as the y-value for later when training the model. I also added a checknan function, which prints the rows where a certain column has a nan value, as well as a fillnan function which fills nan values with a method of your choice: mean, median, or with the values of another column.

After checking for nans in all the columns, these are the columns which turned up:
- Age
- Annual Income
- Marital Status
- Number of Dependents
- Occupation
- Health Score
- Previous Claims
- Vehicle Age
- Credit Score
- Insurance Duration
- Customer Feedback

In the next 11 chapters, we will be tackling each column. 1 column per chapter, esketit.

## Version 3:
For age, the data has a pretty even distribution (see Fig 1 below), so it wont really matter whether we use mean or median for fillna, both values are rather close anyway. I'll just fillna with mean because thats more familiar to most people compared to median.

Fig 1:

![image](https://github.com/user-attachments/assets/19feaa09-d838-4f44-8372-7c38a9f88971)

## Version 4:
Most income distributions are skewed right with a few high income individuals and many low income individuals, and the income data we are working with is no different (see Fig 2 below):

Fig 2:

![image](https://github.com/user-attachments/assets/81ca8f5b-2d8d-4feb-b989-9564460f9abe)

For this ill be using the median to fillna. The mean in income distributions tend to be much higher than the median due to the right skewing, hence i feel median is a better representation of the "average person's" income.

## Version 5:

