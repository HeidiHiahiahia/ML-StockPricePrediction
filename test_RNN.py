# import required packages
import tensorflow as tf
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
def main():
	model = tf.keras.models.load_model('./models/b2luong_s22ye_RNN_model') # Load saved model

	test_data_RNN=pd.read_csv(r'./data/test_data_RNN.csv') #Load saved data set


	test_x_RNN=test_data_RNN.drop('Target',axis=1)
	test_y_RNN=test_data_RNN.drop(['Volume1','Volume2','Volume3','Open1','Open2','Open3','High1','High2','High3','Low1','Low2','Low3'],axis=1)
	

	loss, acc = model.evaluate(test_x_RNN, test_y_RNN, verbose=0)

	print('Restored model, loss: {:5.2f}%'.format(100 * loss))

if __name__ == "__main__":
	# 1. Load your saved model

	# 2. Load your testing data
	main()
	# 3. Run prediction on the test data and output required plot and loss