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

## Version 3: Age
For age, the data has a pretty even distribution (see Fig 1 below), so it wont really matter whether we use mean or median for fillna, both values are rather close anyway. I'll just fillna with mean because thats more familiar to most people compared to median.

Fig 1:

![image](https://github.com/user-attachments/assets/19feaa09-d838-4f44-8372-7c38a9f88971)

## Version 4: Annual Income
Most income distributions are skewed right with a few high income individuals and many low income individuals, and the income data we are working with is no different (see Fig 2 below):

Fig 2:

![image](https://github.com/user-attachments/assets/81ca8f5b-2d8d-4feb-b989-9564460f9abe)

For this ill be using the median to fillna. The mean in income distributions tend to be much higher than the median due to the right skewing (32745.2 and 23911.0 respectively, hence i feel median is a better representation of the "average person's" income.

## Version 5: Marital Status
In the 'Marital Status' column, there are 18529 NaN values, which sounds like alot but only makes up 0.015 (1.5%) of the total training dataset. To fix this, i will just replace all with 'unknown' to play it safe as im not confident the nan values follow the mode. I might change this to mode later on for experimenting and see if it increases the accuracy score of the model. Sticking with the safe option for now.

## Version 6: Number of Dependents
Gonna fillnan with the mean for this, it doesnt matter much anyway as both the mean and median are 2.0 when rounded off to 1dp.

## Version 7: Occupation
This column shows whether one is employed, unemployed, or self-employed. Employment is a big factor in the cost of insurance for many reasons. For example, your route to work may be a dangerous one so car and life insurance may be more expensive, or your insurance comes from your company meaning cheaper costs as you are buying as part of a bulk purchase of insurance with other employees. Overall, employment tends to result in cheaper insurance premiums, especially since its a sign of responsibility in risk assessment in the eyes of insurance companies. I will fillnan with the mode as of now, but i may completely remove this column in a future column as a large proportion of data is missing (30%), even though its an important factor.

## Version 8: Health Score
With a reasonably normal distribution, this would do well with a mean fillnan. median works too, both values are very close to each other

## Version 9: Previous Claims

## Version 10: Vehicle Age

## Version 11: Credit Score

## Version 12: Insurance Duration

## Version 13: Customer Feedback
