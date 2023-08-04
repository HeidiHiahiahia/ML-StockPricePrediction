# Inspiration from:
# https://towardsdatascience.com/a-complete-step-by-step-tutorial-on-sentiment-analysis-in-keras-and-tensorflow-ea420cc8913f#:~:text=Sentiment%20analysis%20is%20one%20of,most%20popular%20deep%20learning%20library.
# https://www.youtube.com/watch?v=hprBCp_UJN0

import keras
import tensorflow as tf
import numpy as np
from matplotlib import pyplot
import pandas as pd
import csv
import re

def parse_TrainingBOW(file):
	'''
	Assuming we will always have the BOW.feat files we can parse them to get rating and 'encoded' reviews
	'''
	with open(file, 'r') as f:
		lines= f.readlines()
	
	data=[]
	ratings=[]
	# count = 0
	for line in lines:

		rating=int(line.split(maxsplit=1)[0])
		if rating <5: #if rating is less than 5 stars, its negative review
			rating=[0]
		else:
			rating=[1]

		st=line.split(maxsplit=1)[1] #take everything else other than rating
		st=re.sub('(:[0-9]+)+',' ',st) #remove the frequency of each word
		encoded_review=[int(index) for index in st.split()] # convert from str to int
		data.append(encoded_review) # add each parsed line into a 2d array, each row being a new input for model
		ratings.append(rating)
	
	# SAVE RESULTS IN .CSV FOR EASIER ACCESS IF NEEDED
	with open('./data/aclImdb/train/train_data.csv', 'w') as f:
		# create the csv writer
		writer = csv.writer(f)
		# write a row to the csv file
		writer.writerows(data)
	with open('./data/aclImdb/train/train_labels.csv', 'w') as f:
		# create the csv writer
		writer = csv.writer(f)
		# write a row to the csv file
		writer.writerows(ratings)
	return data,ratings

def parse_TestingBOW(file):
  with open(file, 'r') as f:
    lines= f.readlines()
  
  data=[]
  ratings=[]
  # count = 0
  for line in lines:

    rating=int(line.split(maxsplit=1)[0])
    if rating <5: #if rating is less than 5 stars, its negative review
      rating=[0]
    else:
      rating=[1]

    st=line.split(maxsplit=1)[1] 
    st=re.sub('(:[0-9]+)+',' ',st) #remove the frequency of each word
    encoded_review=[int(index) for index in st.split()]
    data.append(encoded_review)
    ratings.append(rating)
  
  with open('./data/aclImdb/test/Test_data.csv', 'w') as f:
      # create the csv writer
      writer = csv.writer(f)
      # write a row to the csv file
      writer.writerows(data)
  with open('./data/aclImdb/test/Test_labels.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    writer.writerows(ratings)
  return data,ratings

class NLP():
	def __init__(self, hidden_nodes=16, num_of_hidden_layers=1, embedd_out=16, input_length=500):

		self.hidden_nodes=hidden_nodes
		self.embedd_out=embedd_out
		self.input_length=input_length
		self.num_of_hidden_layers=num_of_hidden_layers

		self.model=keras.Sequential([keras.layers.Embedding(100000,self.embedd_out,input_length=self.input_length),
                        keras.layers.GlobalAveragePooling1D()
                        # keras.layers.Dense(self.hidden_nodes,activation='relu'),
                        ])
		# ADD HIDDEN LAYERS				
		for i in range(self.num_of_hidden_layers):
			self.model.add(keras.layers.Dense(self.hidden_nodes,activation='relu'),)
		self.model.add(keras.layers.Dense(1,activation='sigmoid'))


		self.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

	def learn(self,training_Data,training_Labels,testing_Data,testing_Labels):
		self.history=self.model.fit(training_Data,training_Labels,epochs=8,batch_size=50, validation_data=(testing_Data,testing_Labels), verbose=2)


	def save_model(self):
		# self.model.save('./models/b2luong_s22ye_NLP_model'+str(self.input_length)+"_"+str(self.num_of_hidden_layers)+"_"+str(self.hidden_nodes)+"_"+str(self.embedd_out)+"_"+str(self.input_length))
		self.model.save('./models/b2luong_s22ye_NLP_model')

	def save_acc(self):
		pyplot.figure(figsize=(8, 8), dpi=80)
		pyplot.title('Classification Accuracy')
		pyplot.plot(self.history.history['accuracy'], color='blue', label='train')
		pyplot.plot(self.history.history['val_accuracy'], color='red', label='test')
		pyplot.xlabel('Epoch')
		pyplot.ylabel('Accuracy/100')
		pyplot.legend()

		pyplot.savefig("NLP_acc_"+str(self.input_length)+"_"+str(self.num_of_hidden_layers)+"_"+str(self.hidden_nodes)+"_"+str(self.embedd_out)+"_"+str(self.input_length)+".png")



def main():
	# print("reading TrainingBOW")
	training_Data_raw, training_Labels_raw=parse_TrainingBOW('./data/aclImdb/train/labeledBow.feat')
	# print("Done...")
	# print("reading TestingBOW")
	testing_Data_raw, testing_Labels_raw=parse_TestingBOW('./data/aclImdb/test/labeledBow.feat')
	# print("Done...")

	# ENSURE ALL INPUT DATA IS OF THE SAME SIZE AND CORRECT TYPES
	training_Labels=np.array(training_Labels_raw)
	testing_Labels=np.array(testing_Labels_raw)

	

	for hidden_nodes in [16]:
		num_of_hidden_layers=1
		for input_length in [500]:


			training_Data=tf.keras.utils.pad_sequences(training_Data_raw,value=99999,padding='post',truncating='post', maxlen=input_length)
			testing_Data=tf.keras.utils.pad_sequences(testing_Data_raw,value=99999,padding='post',truncating='post', maxlen=input_length)

			network=NLP(hidden_nodes, num_of_hidden_layers ,hidden_nodes, input_length) #hidden_nodes= embedd_out
			network.learn(training_Data,training_Labels,testing_Data,testing_Labels)
			network.save_model()
			network.save_acc()









	# # CREATE MODEL
	# model=keras.Sequential([keras.layers.Embedding(100000,16,input_length=500),
    #                     keras.layers.GlobalAveragePooling1D(),
    #                     keras.layers.Dense(16,activation='relu'),
    #                     keras.layers.Dense(1,activation='sigmoid')])
	
	# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
	# print("Learning...")
	# history=model.fit(training_Data,training_Labels,epochs=8,batch_size=50, validation_data=(testing_Data,testing_Labels), verbose=2)
	# print("Done...")

	# model.save('./models/b2luong_s22ye_NLP')

	# pyplot.figure(figsize=(8, 8), dpi=80)
	# pyplot.title('Classification Accuracy')
	# pyplot.plot(history.history['accuracy'], color='blue', label='train')
	# pyplot.plot(history.history['val_accuracy'], color='red', label='test')
	# pyplot.xlabel('Epoch')
	# pyplot.ylabel('Accuracy/100')
	# pyplot.legend()

	
	# pyplot.savefig("NLP_acc.png")

if __name__ == "__main__": 
	# 1. load your training data
	main()
	# 2. Train your network
	# 		Make sure to print your training loss and accuracy within training to show progress
	# 		Make sure you print the final training accuracy

	# 3. Save your model