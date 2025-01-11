
![image](https://github.com/user-attachments/assets/9dd1c04e-9415-4bf1-8fca-9db983a1d3b0)

- 🗪 Feel free to telegram me [@milkbottledude](https://t.me/milkbottledude) if you have any questions, or just want to chat :)

## Overview 🔍

This project aims to predict insurance premiums based on various factors, such as but not limited to:
- Age 🎂
- Occupation 💼
- Marital Status 💍
- Income 💵
  
The full list of factors can be found in the [sample.csv](./sample.csv) file, the first 100 rows of the test dataset used for the competition. Here is the link to the [official Kaggle insurance competition page](https://www.kaggle.com/competitions/playground-series-s4e12/overview), although you need to join the competition with a valid account to view the rest of the datasets.

Here i experiment with 3 different machine learning models:
- Random Forest Regression model 🌳
  - takes the average prediction from multiple decision trees, good for non-linear relationships 
- Linear Regression model 📈
  - plain and simple model that assumes a linear relationship between x variables and y
- Tensorflow Neural Network model 🧠
  - similar to RFR models in that it handles non linearity well, different as it better handles large datasets but is harder to interpret and set up 

I'll be documenting my machine learning process below, which will include:
- how i set up the machine learning models 🔧
- prepping existing and creating new data for the models 📊
- how the models performed and the final results 🏆
- the frustrations experienced and lessons learnt 😤

and much more. Similar to the way [Kaggle-Depressed_Machine_Learning](https://github.com/milkbottledude/Kaggle-Depressed_Machine_Learning) is formatted, another competition repository of mine you can check out.

Once again, feel free to skip to any chapters or versions that interest you. Its a long report, and its perfectly understandable if you want to skim and gloss over certain portions 😊.

## Table of Content
Chapter 1: Data Cleaning (Versions 1-15)
- V1: A New Start
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
- V19: Seeing how a normal Decision Tree Regressor fares
- V20: Trying RandomForestRegressor model again
- V21: TensorFlow Neural Network model
- V22: Configuring TF NN model hyperparams
- V23: Configuring data for NN model
- V24: First NN model submission (big improvement!)
- V25: Adjusting hyperparameter - Neuron number
- V26: Adjusting hyperparameter - Batch size
- V27: Adjusting batch size pt 2
- V28: Adjusting hyperparameter - Layer number
- V29: Scaling data
- V30: Adjusting hyperparameter - Epoch number
- V31: Adjusting number of epochs pt 2
- V32: Experimenting with data - datetime values
- V33: Experimenting with datetime values pt 2

Chapter 3: Conclusion


## 📚 Documentation

## Chapter 1 - Data Cleaning
### Version 1: A New Start
Brand new Kaggle notebook, haven't edited anything yet, this is just a standard notebook you get when you create a new notebook in Kaggle. It contains 1 single cell which imports the necessary training and test datasets (train.csv and test.csv), as well as some default packages necessary for machine learning such as numpy and pandas.

### Version 2: Identifying NaN columns
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

### Version 3: Age
For age, the data has a pretty even distribution (see Fig 1 below), so it wont really matter whether we use mean or median for fillna, both values are rather close anyway. I'll just fillna with mean because thats more familiar to most people compared to median.

Fig 1:

![image](https://github.com/user-attachments/assets/19feaa09-d838-4f44-8372-7c38a9f88971)

### Version 4: Annual Income
Most income distributions are skewed right with a few high income individuals and many low income individuals, and the income data we are working with is no different (see Fig 2 below):

Fig 2:

![image](https://github.com/user-attachments/assets/81ca8f5b-2d8d-4feb-b989-9564460f9abe)

For this ill be using the median to fillna. The mean in income distributions tend to be much higher than the median due to the right skewing (32745.2 and 23911.0 respectively, hence i feel median is a better representation of the "average person's" income.

### Version 5: Marital Status
In the 'Marital Status' column, there are 18529 NaN values, which sounds like alot but only makes up 0.015 (1.5%) of the total training dataset. To fix this, i will just replace all with 'unknown' to play it safe as im not confident the nan values follow the mode. I might change this to mode later on for experimenting and see if it increases the accuracy score of the model. Sticking with the safe option for now.

### Version 6: Number of Dependents
Gonna fillnan with the mean for this, it doesnt matter much anyway as both the mean and median are 2.0 when rounded off to 1dp.

### Version 7: Occupation
This column shows whether one is employed, unemployed, or self-employed. Employment is a big factor in the cost of insurance for many reasons. For example, your route to work may be a dangerous one so car and life insurance may be more expensive, or your insurance comes from your company meaning cheaper costs as you are buying as part of a bulk purchase of insurance with other employees. Overall, employment tends to result in cheaper insurance premiums, especially since its a sign of responsibility in risk assessment in the eyes of insurance companies. I will fillnan with the mode as of now, but i may completely remove this column in a future column as a large proportion of data is missing (30%), even though its an important factor.

### Version 8: Health Score
With a reasonably normal distribution, this would do well with a mean fillnan. median works too, both values are very close to each other.

### Version 9: Previous Claims
Using mode for this as there arent a wide range of values, which is 0. My reasoning is that its quite rare for people to have filed an insurance claim before. Me personally, i dont know anyone who has used their insurance before, but that could be because i have little friends

### Version 10: Vehicle Age
Distribution for this column is fairly even (see Fig 3 below), mean and median values dont defer by much, but ill just use median cuz i dont have to round off to a whole number

Fig 3:

![image](https://github.com/user-attachments/assets/e24d0623-3f61-466c-8c04-abd3d347b9b9)


### Version 11: Credit Score
A distribution with no ridiculous outliers (see Fig 4 below), both mean and median values are extremely similar, will fillna with median.

Fig 4:

![image](https://github.com/user-attachments/assets/6f45d916-284e-47ad-b62f-2fd922b7e52d)

### Version 12: Insurance Duration
Fig 5:

![image](https://github.com/user-attachments/assets/d5436670-8d87-4299-9571-ef2f506b7c5f)

Median. Nuff said.

### Version 13: Customer Feedback
Not a wide range of values for this column (poor, average, good). Just gonna use mode for fillna, which is the value 'average'

### Version 14: Processing 'Policy Start Date' column
As the first model i plan to use, randomforestregressor, cant handle datetime values, i have to process the data and change its format for something more 'digestible' for the model. My plan is to simply split the values into 2 separate columns, 'start year' and 'start month'. For the start year, i will leave it as it is as the randomforest models are good at managing such unscaled features without much trouble, and there is a general trend of insurance premiums increasing as the years pass (Fig 6 below shows gross insurance premiums against time). Contrary to this, there is no valid or apparent linear trend between month value and insurance price, so i will be converting the month values into qualitative values (1 to Jan, 2 to Feb, etc). For the rest of the elements in the datetime value (day and time), ill discard them as their small significance compared to year and month make them unlikely clues to find patterns and trends.

Fig 6: (credits to statista: https://www.statista.com/outlook/fmo/insurances/worldwide)

![image](https://github.com/user-attachments/assets/e15a2650-f449-41ea-82c3-b6f95622a60c)

### Version 15: Making the Kaggle notebook run smoother 
As expected of a training dataset with 1.2 million rows, it exceeded the ram limit of the kaggle notebook and crashed it (Fig 7). i want to train the model within the limits of the ram, but i also dont want to waste data by not using it. I also tidied up the fillnan code that was written in the earlier versions so that its slightly less of an eyesore.

Fig 7:

![image](https://github.com/user-attachments/assets/6ba7d107-588a-4348-9b54-f7b956d84942)


To fix the ram problem, ill be changing up the values in the columns 'Gender', 'Customer Feedback', and 'Smoking Status' to reduce computational stress. ill just show the mapping dictionaries for the 3 columns below.

gendermap = {'Male': 1, 'Female': 0}

feedbackmap = {'Poor': 1, 'Average': 2, 'Good': 3}

smokingmap = {'Yes': 1, 'No': 0}

This way, unnecessary columns wont be created. Originally 2 separate columns for gender, 3 for feedback and 2 for smoking status, now just 1 for each variable. Doing that for 1.2m rows should do the trick. Also converting the feedback string values to numerical values shouldnt pose an issue for the model even though its 'though process' moulds to this thinking that good > average > poor (3 > 2 > 1).

## Chapter 2: Machine Learning model(s) configuration
### Version 16: Training RandomForestRegressor model
Set up the RandomForestRegressor model as well as a simple function that returns the RMSLE when u pass the predicted and actual y values from the mocktest data. Got a RMSLE value of 1.16(2dp) from a stock rfr model with no custom hyperparams except random_state. The top score on the leaderboard is currently 1.03(2dp), so we have work to do. For now ill submit it and see the public RMSLE score.

### Version 17: Using Linear Regression
Submission couldnt go thru as the test data with 800k rows couldnt be predicted, my laptop sounds like a boeing rn. Prob cuz the model had to fit 720k training data from the train_test_split and predict 480k mock test data, before predicting 800k rows from the real test dataset. Nothing like the depression dataset which consisted of only 141k total training rows and 93.8k test rows. Switching to a more simpler LinearRegressor model, but will defo use a more in-depth machine learning model in the future, i have a few in mind. LinearRegressor is a simple model but did better than i thought, getting a RMSLE score of 1.17 (2dp). But when i try to predict with the test dataset, i keep running out of damn ram. I hope to find the problem soon, but rn im stumped.

### Version 18: Finally found the problem
The mistake i made is that I split up the 'Policy Start Date' into year and month columns for the train_data but did not for the test_data, resulting in the test_data getting dummies for 158776 unique datetime values, basically creating over 15k more columns and hence using up crazy ram. I found out when i printed out all unique values in each column for both datasets, and noticed the large number in a row where a large number should not be present (see Fig 8 below). A careless mistake on my part, i apologise. But now we can now finally find out where we stand on the public leaderboard.

Fig 8:

![image](https://github.com/user-attachments/assets/37f1e9de-57c2-4939-94e2-3ca91ccf2665)

Edit: First submission RMSLE - 1.16532, Leaderboard placing - 578/643 (10th percentile)

### Version 19 & 19.1: Decision Tree regressor
Trying out a DecisionTreeRegressor model. On the mock test data, it didnt do as well and got an RMSLE score of 1.52. But if theres one thing i learned from the previous competition (predicting depression), its that the mock test data score could sometimes be very different from the actual score, although granted this kind of occurence is rare. Only one way to find out.

Edit: public RMSLE score - 1.51244 

### Version 20 & 20.1 & 20.1.1: RandomForestRegressor
I'm feeling patient today so im trying this model again, but its going to take a very long time.

Edit: Took a long time (Fig 9), but it was worth it. Got our best public RMSLE score so far, 1.1535. An improvement compared to previous best (1.16532) with the basic linear regression model.

Fig 9:

![image](https://github.com/user-attachments/assets/29e21bd5-7d58-47bf-a9b8-dab2042d38a5)

New leaderboard position - 562/678 (17th percentile, small improvement)

### Version 21: Switching it up with TensorFlow Neural Network 
To start, i instantiated a Keras sequential model, fairly basic with just 3 layers including the input and output layer. Set the neurons in a decreasing manner starting from 128, but this is just the baseline and not the final neuron values. I may also change the layers depending on how the model is fitting. 

### Version 22: Configuring the model further
Here are the hyperparameters ill be using for model.compile():

optimizer: 'adam', im not super familiar with this optimizer if im being honest, but its the default optimizer and works for both classification and regression problems.

loss: 'mean_squared_logarithmic_error'. The Kaggle competition evaluates submissions based on RMSLE, but the compile method does not accept RMSLE as a valid loss function. However, it does accept the next best thing, MSLE.

metrics: ['mae', 'rmse']. Taking advantage of the compile method's ability to have multiple metrics, i chose mae as my primary metric as its a simple way to see the average difference between predicted and actual values. Supplementing the mae is the rmse as it informs us of the presence of large errors, which mae by itself doesnt tell us.

## Version 23: Feeding data into NN model
Before passing the datasets, i gotta convert them from pandas dataframes to numpy arrays. After that, we can train the model using the fit method. For the hyperparameters, starting with 10 for number of epochs. For batch_size, i decided to go with 100. I feel its a value thats not too big that the model overfits, but also not too small that it takes forever all the rows to be trained on. These values are just starter values and ill be adjusting them in the future if needed, depending on whether overfitting or underfitting occurs or if the time taken for training is too long.

## Version 24: Training the model
As the train.csv given to us has so many rows (1.2m), i decided to further split up the training data using the validation_split parameter while fitting the model on top of already splitting it once in an earlier version using train_test_split. This is so that i can leave the mocktest data as a 'final test' after the model has run through all the epochs, before finally submitting the model to the competition. I also set number of epochs to 10. Here are training and validation losses (RMSLE) over the different epochs (Fig 10):

Fig 10:

![image](https://github.com/user-attachments/assets/55d45983-3762-4dfb-b2ee-2d4cf4411735)

As you can see, there is a sharp drop in training RMSLE loss from the 1st epoch to the 2nd, but a only a small drop in the validation loss, which might be a sign of overfitting. The training loss continues to drop slowly but surely over the epochs, while the validation loss stagnates for a while before eventually dropping to 1.16 at the end. 

Before submitting, lets give the model one final test on our mocktest data. RMSLE for this unseen mocktest data was 1.17 (2dp). Overall not a bad first fit, but there is still room for improvement. For now lets submit it first and see whats the public RMSLE score. 

Edit: public RMSLE score - 1.07498, new leaderboard position - 427/929 (54th percentile, big jump)

## Version 25: Adjusting neuron(unit) number.
A very pleasant surpise to have such a huge improvement in score after using a relatively simple and default neural network model. But as mentioned in the previous version, there are still problems that need to be ironed out. One of which might be overfitting, which is indicated by the rapid reduction of training loss which contrasts to the miniscule dip in validation loss in the earlier epochs. This could be due to 3 reasons: 1) Too many neurons in the layers, aka model is too wide, 2) too many layers, aka model is too deep, or 3) batch_size value is too large during training. These three factors lead to the model learning incorrect patterns that have nothing to do with insurance premiums instead of fitting to general trends.

Firstly ill reduce the number of neurons in the first dense layer from 128 to 64, and from 64 to 32 in the 2nd layer. Now lets look at the new loss values over the different epochs (Fig 11).

Fig 11:

![image](https://github.com/user-attachments/assets/e0ef0404-4aef-4ec9-869d-725e92475826)

Not much difference in training loss, and basically identical validation loss and mocktest data loss (still 1.17 (2dp)) when neuron number was reduced. But there was no big decrease in training loss for the earlier epochs, which may mean that this is the best the model can do with the data given to it. As i still have a few submissions available for today, ill submit this version with the reduced neuron number and see if theres any change in the public score.

Edit: public RMSLE score - 1.07607

# Version 26: Adjusting training batch_size
Lets see if reducing the batch size from 100 to 50 will help (Fig 12).

Fig 12:

![image](https://github.com/user-attachments/assets/9c1e4b31-0c8b-4759-a1cc-b92efd614144)

Performance worsened slightly surprisingly for the val loss (lowest value was 1.17 instead of 1.16 in previous versions) and mocktest loss (1.18 compared to 1.17 previously). Correspondingly, public RMSLE score also was slightly worse. I had thought overfitting was the problem so reducing batch size to introduce noise would be a viable solution, but apparently not. Back to the drawing board.

Edit: public RMSLE score - 1.07710

## Version 27: Adjusting training batch_size (again)
It could be that the big difference in training loss between the 1st and 2nd epoch was because it was the first time seeing the training data for the 1st epoch hence it had such a huge loss compared to 2nd epoch, and not because of overfitting like i had suspected in Version 24 and 25. In which case reducing batch size in Version 26 might have been a mistake as it introduced unnecessary noise that can mess with weight updates. As of right now im not sure if my model is overfitting or underfitting, so ill be experimenting abit. In this version ill try a higher batch_size, starting with 125. Heres the results of that (Fig 13).

Fig 13:

![image](https://github.com/user-attachments/assets/0bf14ec5-e81c-49c5-94bc-de625f5d48ed)

After that i increased batch_size further to 150 (Fig 14):

Figi 14:

![image](https://github.com/user-attachments/assets/57a4488d-b822-4f62-a958-3e9c0f63c78a)

Training and validation loss decreased when batch_size was increased from 100 to 125 but increased when batch_size went further up to 150. Lets submit this version with batch_size 125 and see how the public score changes.

Edit: public RMSLE score - 1.07663


## Version 28: Adjusting number of layers
In this version i add an extra layer just before the output layer in the keras model and slapped it with 32 neurons and the same activation function as the other hidden layers (relu). If val loss increases, that means the model is overfitting and no extra layer is needed. However, the results (Fig 15) are very similar to not having an extra layer (see Fig 10), in fact its slightly better with its last epoch having a val loss of 1.65 compared to 1.66 in Fig 10. This is a good sign that the extra layer is not causing the model to overfit. Also its loss when tested with the mocktest data is also the same at 1.16 (2dp).

Fig 15:

![image](https://github.com/user-attachments/assets/aec55a63-957f-4a8f-a32e-d527f64a1c29)

Edit: public RMSLE score - 1.07813


## Version 29: Scaling data
For the previous models like forest classifiers (which do not require scaling) and linear regression (which benefit very slightly or not at all from scaling), scaling is not really necessary. But neural networks are trained in such a way that make them sensitive to the scale of the variables, so in this version im just going to apply some minmax scaling to all the input data. Reverting back to just 4 layers since adding another didnt improve public score. After scaling and training, the results are super identical to Fig 10, the val loss for the last few epochs differing by only around 0.0001. However, when fitted with the entire training dataset (all 1.2m rows), the difference starts to show. Below (Fig 16) shows the results when the input is not scaled (from Version 24), and Fig 17 shows when the input is scaled.

Fig 16 (unscaled):

![image](https://github.com/user-attachments/assets/6b74b222-4fda-4448-bce4-9bce70c4e539)

Fig 17 (scaled):

![image](https://github.com/user-attachments/assets/e5a208de-c5b1-411a-9d29-707a714bb9f4)

As you can see the loss in the final epoch is visibily greater when the data is unscaled at 1.1618 compared to when the data is scaled, which is 1.1517. I only hope that this is also reflected in the public score.

Edit: public RMSLE score - 1.0730, new leaderboard position - 480/1055 (55th percentile) 

only a single percentile increase but LESGOOOOO finally an improvement. Funny how after all that tuning, the thing that got the score to improve was some damn scaling.

## Version 30: Adjusting number of epochs
Now that we have tinkered with the layers and neurons, its time to shift our focus to epoch number, which is basically the number of times the model 'goes through' the training data fed to it. Kind of like when you go through a textbook before going into a test. However, unlike when you are revising your textbook, looking through the data too many times can be bad for the model, so now we are going to try and find the optimal epoch for our dataset. Since we started using the nn tensorflow model (Version 21), the epoch number has been set at a nice 10. And from the 9th to the 10th epoch, the losses are still decreasing, so theres a possibility that theres a greater epoch number with an even lower loss. To find this magic epoch number, we need to look at a wide range of epochs beyond 10. 

To visualise how the RMSLE varies with epoch number, we are going to set the epoch number at a high number that should have a high probability of including the optimal epoch number, say 40. Then we are going to access all the loss values using the code "history.history['val_loss']" and plot them against epoch number on a simple line graph.

Edit: ok apparently the optimum epoch number is greater than 40 because leading up to the 40th epoch, the loss is still going down. Thats great, that means this model has the potential to predict with a much lower loss than we thought, so now im going to increase the epoch number from 40 to 75.

Fig 18 (val_loss and train_loss against epochs, 75 epochs):

![image](https://github.com/user-attachments/assets/f757507e-569e-4410-b78e-bb01623c7733)

Looks like the loss is still decreasing even at the last few epochs, which is good news. Lets jump to 100 epochs and see if the loss becomes stagnant.

Fig 19(val_loss and train_loss against epochs, 100 epochs):

![image](https://github.com/user-attachments/assets/e3361763-48db-4ba2-a1d0-74f3b7661bf7)
 
As you can see the val_loss continues to decrease (highlighted in yellow) up until epoch 93, after which it levels out and hovers around 1.503 and 1.504 (highlighted in green).

Fig 20(actual val_loss values at last few epochs near to 100):

![image](https://github.com/user-attachments/assets/fa66ffb2-3126-43ad-bbb4-d9f53cc496aa)

For now im just going to submit this 100 epoch version and see how well it does.

Edit: public RMSLE score - 1.06686, new leaderboard position - 496/1127 (56th percentile)

Another 1 percentile increase, but we take wins no matter how small. First time breaking through RMSLE of 1.07 as well, im happy.

## Version 31 and 31.1: Adjusting number of epochs (again)
Increasing epoch number to 200 to really make sure beyond doubt that the optimal epoch number was covered. You might think this is excessive, but previously when getting ready to submit the 100 epoch version and fitting the entire training dataset (cuz i only fitted about 70% of the training set earlier leaving 30% for validation), i noticed even at 100 epochs that the loss(when fitting the entire training set not the train_test_split) was still reducing. Granted it was the train_loss but the rate it was reducing was still relatively large, which leads me to think that the val loss might be even lower with an epoch number greater than 100. 

To really see how well the entire training data is fitting at the end, ive set the validation_split at 0.05. I set it really small to use as much training data as possible for fitting the final model WHILE still having an indication of where the model starts to overfit and the val_loss stops decreasing. 

Fig 21(val loss against epochs for final fit using 0.05 of training data as val data):

![image](https://github.com/user-attachments/assets/b992edcf-d807-444d-b479-cbf08db43496)

The loss in Fig 21 above starts to stop decreasing notably at around 150 epochs, take note of the different y-axis scale. This proves that the model does continue to improve past 100 epochs, but is that also the case for the actual test data though. In theory yes, but there have been many times when loss was decreasing during training but higher than expected in the public RMSLE ill do another submission with 150 epochs in this version and see if the public score increases.

Edit: public RMSLE score - 1.06437, new leaderboard position - 578/1329 (57th percentile)

To not waste your time reading how i trial and error different epoch values, i went ahead and tried submitting with 200 and 250 epochs. Here are the results -

200 epochs: public RMSLE score - 1.06389, new leaderboard position - 568/1348 (58th percentile)

250 epochs: public RMSLE score - 1.06533

At 200 the RMSLE decreased showing the model was far from its optimum epoch number at 150 epochs, but at 250 there was an increase in RMSLE that is greater than that of 150 epochs, an obvious sign of overfitting.

## Version 32, 32.2 & 32.3: Fiddling with datetime values pt 1
I've messed with hyperparameters enough, so in this version i want to focus more on the actual variables, one of which is the datetime column. As you know i created called get_dummies to create dummy columns for the month portion of the datetime values (Version 14), essentially creating 12 columns for each month. However, i recently learned that having all 12 introduces multicollinearity. For example, if you see that the value for every column representing the months January to November is 'False', you automatically know that December is 'True'. However i like the month of December, so if i had to get rid of a month it would be January. Also im going to comment out the fitting of the mock training data, which is the fraction of the whole training data that i split using train_test_split, to save time during submission. A single 200 epoch fitting already takes a long time, id rather not do it twice for every submission.

Edit: public RMSLE score - 1.06561 

Not too sure why it did worse, in theory it should have improved because everything was the same as version 31 which was the version that got the best public score so far. Edit: i think it may have something to do with the 'keras.utils.set_random_seed(0)' code, which by right is supposed to make all random operations that occur during the creation and training of the model (such as setting of weights etc) constant whenever the code is run for result reproducibility, but apparently this is an outdated way to set random seed, a fault on my part. Future version will use the more recent 'tf.random.set_seed(0)' to set constant random seed. To test this ill be submitting an exact replica of Version 31 and seeing if the results are still 1.06389. Edit: it was not, i got 1.06448. Apparently its quite common to have different results for the same code in Kaggle competitions, there are numerous complaints on reddit from people who have experienced the same thing. Oh well, close enough i guess.

## Version 33: Fiddling with datetime values pt 2
Now i will be trying out this other cool new data prepping method for datetime values called 'cyclic encoding', sounds very cool i know. What is unique about the month portion in the datetime values is that they are cyclic, meaning although December is the 12th month and January is the 1st, the 2 months are actually right next to each other even though their numbers are not (12 and 1). This new method is basically just applying both the sin and cos function to the month numbers which accounts for the cyclic nature of the data. This may not beat our all time best score of 1.06389 due to the random seed problem which stopped us from being able to reproduce results, but it should beat 1.06448, which is the score we got after changing the random seed line of code.

Edit: public RMSLE score - 1.06547

Always remember to change your neural network model's input layer shape after feature engineering. Dont be like me who left the notebook running for 2 hours, just to come back and see an error message at the fitting stage. 

## Chapter 3 - Conclusion

And thats the end of the competition, here is my final standing with the rest of the test data used by Kaggle to compute the final leaderboard at the end of the 30 day competition duration:

Final public RMSLE score - 1.06612, Final leaderboard position - 1074/2392 (55th percentile)

Once again, this was fun. Big learning point from this project was definitely neural networks, and learning it was certainly an ordeal. Not just tuning and feature engineering, but just prepping the data to be put into the model and learning HOW to put the data into the model, creating layers etc was also difficult to pick up. But totally worth it when i saw the RMSLE score improve by so much and the leaderboard position shoot up. Neural networks definitely have a big advantage over traditional regression or tree classifier models in many ways, particularly handling non linearity in the data. So im very glad to have been given the opportunity to learn and practice this type of machine learning model on a dataset as large and interesting as this insurance one. Im looking forward to the next project where i can test out my new neural network skills as well as improve my tuning of the model. 

If you have read this far, thank you so much. I hope this was as informative and enjoyable to you as it was for me. Have a good day, dont forget to drink water. 

ok byee :) 👋
