#Installing necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from matplotlib import pyplot

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


def main():
	#Read in provided stock data
	stock0=pd.read_csv(r'./data/q2_dataset.csv')

	#Reversing rows so the data starts from the earliest dates
	stock0 = stock0.loc[::-1]

	#Create new stock dataset stock1 that has 12 Parameters +1 Target
	stock1=np.zeros((1256,13))

	#Add values to new stock dataset

	for i in range(1256):
		#add Volume data for 3 days
		stock1[i][0]=stock0.loc[i][2]
		stock1[i][1]=stock0.loc[i+1][2]
		stock1[i][2]=stock0.loc[i+2][2]
		#add Open data for 3 days
		stock1[i][3]=stock0.loc[i][3]
		stock1[i][4]=stock0.loc[i+1][3]
		stock1[i][5]=stock0.loc[i+2][3]
		#add High data for 3 days
		stock1[i][6]=stock0.loc[i][4]
		stock1[i][7]=stock0.loc[i+1][4]
		stock1[i][8]=stock0.loc[i+2][4]
		#add Low data for 3 days
		stock1[i][9]=stock0.loc[i][5]
		stock1[i][10]=stock0.loc[i+1][5]
		stock1[i][11]=stock0.loc[i+2][5]
		#add Target Open data for the 4th day
		stock1[i][12]=stock0.loc[i+3][3]
	#Define columns and dataframe
	stock1_col=['Volume1','Volume2','Volume3','Open1','Open2','Open3','High1','High2','High3','Low1','Low2','Low3','Target']
	df=pd.DataFrame(stock1[:-2,:],columns=stock1_col)

	#randomize dataset by rows
	df_s = df.sample(frac=1)

	#Split data to training dataset and test dataset 70% &30%
	train_data_RNN,test_data_RNN = train_test_split(df, test_size=0.3, random_state=25)

	#Write training dataset and test dataset to .csv
	train_data_RNN.to_csv(r'./data/train_data_RNN.csv', index = False, header=True)
	test_data_RNN.to_csv(r'./data/test_data_RNN.csv', index = False, header=True)


	#Split traning and test data correspondingly to x and y files
	train_data_RNN.shape
	train_x_RNN=train_data_RNN.drop('Target',axis=1)
	train_y_RNN=train_data_RNN.drop(['Volume1','Volume2','Volume3','Open1','Open2','Open3','High1','High2','High3','Low1','Low2','Low3'],axis=1)
	test_x_RNN=test_data_RNN.drop('Target',axis=1)
	test_y_RNN=test_data_RNN.drop(['Volume1','Volume2','Volume3','Open1','Open2','Open3','High1','High2','High3','Low1','Low2','Low3'],axis=1)
	test_y_RNN_t=test_y_RNN.reset_index(drop=True)

	#Building LSTM model
	regressor = Sequential() #initializing NN

	regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (train_x_RNN.shape[1], 1)))
	regressor.add(Dropout(0.2)) #adding dropout layers that prevent overfitting

	regressor.add(LSTM(units = 50, return_sequences = True))
	regressor.add(Dropout(0.2))

	regressor.add(LSTM(units = 50, return_sequences = True))
	regressor.add(Dropout(0.2))

	regressor.add(LSTM(units = 50))
	regressor.add(Dropout(0.2))

	regressor.add(Dense(units = 1)) #add densely connected neural network layer

	# regressor.summary()


	#Training the model 
	regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

	regressor.fit(train_x_RNN, train_y_RNN, epochs = 600, batch_size = 32, verbose=2)

	regressor.save('./models/b2luong_s22ye_RNN_model')

	predicted_stock_price = regressor.predict(test_x_RNN, verbose=2)
	e_stock_price = regressor.evaluate(test_x_RNN,test_y_RNN, verbose=2)

	# loss_test = regressor.evaluate(test_x_RNN, verbose=2)
	print('The loss on test data is', e_stock_price[0])

	import matplotlib.pyplot as plt
	plt.figure(figsize=(15,5))
	plt.plot(test_y_RNN_t, color = 'black', label = 'Test Data')
	plt.plot(predicted_stock_price, color = 'green', label = 'Predicted')
	plt.title('Stock Price Prediction vs. Test Data')
	plt.xlabel('Time')
	plt.ylabel(' Stock Price')
	plt.legend()
	plt.savefig("PredictedStocks.png")


if __name__ == "__main__": 
	# 1. load your training data
	main()
	# 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss

	# 3. Save your model