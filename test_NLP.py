import tensorflow as tf
import numpy as np
# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
def main():
	model = tf.keras.models.load_model('./models/b2luong_s22ye_NLP_model')

	# model.summary()

	# testing_Data=pd.read_csv('./data/aclImdb/test/Test_data.csv')
	# testing_Labels=pd.read_csv('./data/aclImdb/test/Test_labels.csv')

	with open("./data/aclImdb/test/Test_data.csv", 'r') as test_data:
		# testing_Data=np.genfromtxt(test_data, delimiter=',', filling_values=99999)  #load in the trained weights
		lines= test_data.readlines()
		testing_Data=[]
		for line in lines:
			line=line.strip()
			testing_Data.append( [ int(element) for element in line.split(',')])
		testing_Data=np.array(testing_Data)

	# print(testing_Data)
	with open("./data/aclImdb/test/Test_labels.csv", 'r') as test_data:
		# testing_Data=np.genfromtxt(test_data, delimiter=',', filling_values=99999)  #load in the trained weights
		lines= test_data.readlines()
		testing_Labels=[]
		for line in lines:
			line=line.strip()
			testing_Labels.append( [ int(element) for element in line.split(',')])
		testing_Labels=np.array(testing_Labels)
	# with open("./data/aclImdb/test/Test_labels.csv", 'r') as test_data:
	# 	testing_Labels=np.genfromtxt(test_data, delimiter=',', filling_values=99999)  #load in the trained weights

	testing_Data=tf.keras.utils.pad_sequences(testing_Data,value=99999,padding='post',truncating='post', maxlen=500)
	testing_Labels=np.array(testing_Labels)

	loss, acc = model.evaluate(testing_Data, testing_Labels, verbose=0)

	print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

if __name__ == "__main__": 
	# 1. Load your saved model

	# 2. Load your testing data
	main()

	# 3. Run prediction on the test data and print the test accuracy