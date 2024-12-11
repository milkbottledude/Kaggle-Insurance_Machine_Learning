# Kaggle-Insurance_Machine_Learning
This machine learning project aims to predict insurance premiums based on various factors. I'll be documenting my machine learning process in this repository, similar to the Kaggle-Depressed_Machine_Learning repository. The latter competition took up a lot of time and i got rewarded with only dark eye bags, and im here to do it all over again. Perhaps i should try other hobbies like fishing instead.

Once again, feel free to skip to any chapters or versions that interest you, its a long report and its perfectly understandable if you want to just skim certain portions.

## Contents
Chapter 1: Data Cleaning (Versions 1-15)
- V1: A new start
- V2: Identifying NaN columns
- V3: Tackling NaNs, starting with the 'Age' column
- V4: Annual Income
- V5: Marital Status
- V6: Number of Dependents
- V7: Occupation
- V8: Health Score
- V9: Previous Claims
- V10: Vehicle Age
- V11: Credit Score
- V12: Insurance Duration
- V13: Customer Feedback
- V14: Processing 'Policy Start Date' column
- V15: Making the Kaggle notebook run smoother

Chapter 2: Machine Learning model(s) configuration
- V16: Training RandomForestRegressor model (a problem occured here)
- V17: Resorting to Linear Regression
- V18: Finally found the problem (and resolved it)
- V19: Switching it up with TensorFlow Neural Network 

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
With a reasonably normal distribution, this would do well with a mean fillnan. median works too, both values are very close to each other.

## Version 9: Previous Claims
Using mode for this as there arent a wide range of values, which is 0. My reasoning is that its quite rare for people to have filed an insurance claim before. Me personally, i dont know anyone who has used their insurance before, but that could be because i have little friends

## Version 10: Vehicle Age
Distribution for this column is fairly even (see Fig 3 below), mean and median values dont defer by much, but ill just use median cuz i dont have to round off to a whole number

Fig 3:

![image](https://github.com/user-attachments/assets/e24d0623-3f61-466c-8c04-abd3d347b9b9)


## Version 11: Credit Score
A distribution with no ridiculous outliers (see Fig 4 below), both mean and median values are extremely similar, will fillna with median.

Fig 4:

![image](https://github.com/user-attachments/assets/6f45d916-284e-47ad-b62f-2fd922b7e52d)

## Version 12: Insurance Duration
Fig 5:

![image](https://github.com/user-attachments/assets/d5436670-8d87-4299-9571-ef2f506b7c5f)

Median. Nuff said.

## Version 13: Customer Feedback
Not a wide range of values for this column (poor, average, good). Just gonna use mode for fillna, which is the value 'average'

## Version 14: Processing 'Policy Start Date' column
As the first model i plan to use, randomforestregressor, cant handle datetime values, i have to process the data and change its format for something more 'digestible' for the model. My plan is to simply split the values into 2 separate columns, 'start year' and 'start month'. For the start year, i will leave it as it is as the randomforest models are good at managing such unscaled features without much trouble, and there is a general trend of insurance premiums increasing as the years pass (Fig 6 below shows gross insurance premiums against time). Contrary to this, there is no valid or apparent linear trend between month value and insurance price, so i will be converting the month values into qualitative values (1 to Jan, 2 to Feb, etc). For the rest of the elements in the datetime value (day and time), ill discard them as their small significance compared to year and month make them unlikely clues to find patterns and trends.

Fig 6: (credits to statista: https://www.statista.com/outlook/fmo/insurances/worldwide)

![image](https://github.com/user-attachments/assets/e15a2650-f449-41ea-82c3-b6f95622a60c)

## Version 15: Making the Kaggle notebook run smoother 
As expected of a training dataset with 1.2 million rows, it exceeded the ram limit of the kaggle notebook and crashed it (Fig 7). i want to train the model within the limits of the ram, but i also dont want to waste data by not using it. I also tidied up the fillnan code that was written in the earlier versions so that its slightly less of an eyesore.

Fig 7:

![image](https://github.com/user-attachments/assets/6ba7d107-588a-4348-9b54-f7b956d84942)


To fix the ram problem, ill be changing up the values in the columns 'Gender', 'Customer Feedback', and 'Smoking Status' to reduce computational stress. ill just show the mapping dictionaries for the 3 columns below.

gendermap = {'Male': 1, 'Female': 0}

feedbackmap = {'Poor': 1, 'Average': 2, 'Good': 3}

smokingmap = {'Yes': 1, 'No': 0}

This way, unnecessary columns wont be created. Originally 2 separate columns for gender, 3 for feedback and 2 for smoking status, now just 1 for each variable. Doing that for 1.2m rows should do the trick. Also converting the feedback string values to numerical values shouldnt pose an issue for the model even though its 'though process' moulds to this thinking that good > average > poor (3 > 2 > 1).

# Chapter 2: Machine Learning model(s) configuration

## Version 16: Training RandomForestRegressor model
Set up the RandomForestRegressor model as well as a simple function that returns the RMSLE when u pass the predicted and actual y values from the mocktest data. Got a RMSLE value of 1.16(2dp) from a stock rfr model with no custom hyperparams except random_state. The top score on the leaderboard is currently 1.03(2dp), so we have work to do. For now ill submit it and see the public RMSLE score.

# Version 17: Resorting to Linear Regression
Submission couldnt go thru as the test data with 800k rows couldnt be predicted, my laptop sounds like a boeing rn. Prob cuz the model had to fit 720k training data from the train_test_split and predict 480k mock test data, before predicting 800k rows from the real test dataset. Nothing like the depression dataset which consisted of only 141k total training rows and 93.8k test rows. Switching to a more simpler LinearRegressor model, but will defo use a more in-depth machine learning model in the future, i have a few in mind. LinearRegressor is a simple model but did better than i thought, getting a RMSLE score of 1.17 (2dp). But when i try to predict with the test dataset, i keep running out of damn ram. I hope to find the problem soon, but rn im stumped.

# Version 18: Finally found the problem
The mistake i made is that I split up the 'Policy Start Date' into year and month columns for the train_data but did not for the test_data, resulting in the test_data getting dummies for 158776 unique datetime values, basically creating over 15k more columns and hence using up crazy ram. I found out when i printed out all unique values in each column for both datasets, and noticed the large number in a row where a large number should not be present (see Fig 8 below). A careless mistake on my part, i apologise. But now we can now finally find out where we stand on the public leaderboard.

Fig 8:

![image](https://github.com/user-attachments/assets/37f1e9de-57c2-4939-94e2-3ca91ccf2665)

Edit: First submission RMSLE - 1.16532, Leaderboard placing - 578/643

## Version 19 & 19.1:
Trying out a DecisionTreeRegressor model. On the mock test data, it didnt do as well and got an RMSLE score of 1.52. But if theres one thing i learned from the previous competition (predicting depression), its that the mock test data score could sometimes be very different from the actual score, although granted this kind of occurence is rare. Only one way to find out.

Edit: public RMSLE score - 1.51244 

## Version 20 & 20.1 & 20.1.1: RandomForestRegressor
I'm feeling patient today so im trying this model again, but its going to take a very long time.

Edit: Took a long time (Fig 9), but it was worth it. Got our best public RMSLE score so far, 1.1535. An improvement compared to previous best (1.16532) with the basic linear regression model.

Fig 9:

![image](https://github.com/user-attachments/assets/29e21bd5-7d58-47bf-a9b8-dab2042d38a5)

Current leaderboard position: 562/678

## Version 21: Switching it up with TensorFlow Neural Network 
To start, i instantiated a Keras sequential model, fairly basic with just 3 layers including the input and output layer. Set the neurons in a decreasing manner starting from 128, but this is just the baseline and not the final neuron values. I may also change the layers depending on how the model is fitting. 

## Version 22: Configuring the model further
Here are the hyperparameters ill be using for model.compile():

optimizer: 'adam', im not super familiar with this optimizer if im being honest, but its the default optimizer and works for both classification and regression problems.

loss: 'mean_squared_logarithmic_error'. The Kaggle competition evaluates submissions based on RMSLE, but the compile method does not accept RMSLE as a valid loss function. However, it does accept the next best thing, MSLE.

metrics: ['mae', 'rmse']. Taking advantage of the compile method's ability to have multiple metrics, i chose mae as my primary metric as its a simple way to see the average difference between predicted and actual values. Supplementing the mae is the rmse as it informs us of the presence of large errors, which mae by itself doesnt tell us.

## Version 23: Feeding data into NN model
Before passing the datasets, i gotta convert them from pandas dataframes to numpy arrays. After that, we can train the model using the fit method. For the hyperparameters, starting with 10 for number of epochs. For batch_size, i decided to go with 100. I feel its a value thats not too big that the model overfits, but also not too small that it takes forever all the rows to be trained on. These values are just starter values and ill be adjusting them in the future if needed, depending on whether overfitting or underfitting occurs or if the time taken for training is too long.

## Version 24
