#This is the code for the graded project for course 5: deep learning
#Elias Chavarria-Mora

import os #for changing working direction
import pandas as pd
import numpy as np
np.__version__ #1.26.4
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer #for stemming
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
tf.__version__ #2.3.0, apparently sometimes tensorflow and numpy are incompatible, I need to upgate to np above 1.22 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU 
#These are types of layers for the neural network
from tensorflow.keras.layers import SimpleRNN #this is the recurrent neural network model
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing.sequence import pad_sequences #LOL, este aparentement lo mueven mucho de un
#path a otro, hay que buscarlo
import gc #garbage collection, eliminates unnecessary variables


def tokenize(text):
    return text.split()

snowballStemmer = SnowballStemmer("spanish")
def tokenize_snowball(text):
    return [snowballStemmer.stem(word) for word in text.split()]

stop=stopwords.words('spanish')

os.chdir('../../Elias/1 Serious/Academia/University of Pittsburgh/1 Dissertation/Data') #set working direction

#load the data
ElectoralTweets= './df_FullDiss_V6.csv' 
df=pd.read_csv(ElectoralTweets)
df.head()

#recode the target as boolean
df['Analytic_boolean'] = df['Analytic'].apply(lambda x: 1 if x >= 0 else 0)
#a lambda is a temporary, disposable function: here, it is turnin all non-negative values into 1. 

#check if there are missings
has_missing = df['clean_text'].isnull().any()
print(has_missing)  # Output: True if there are missing values, False otherwise
#how many
number_missings = df['clean_text'].isnull().sum()
print(number_missings)  #just 1, a expected
#drop the missing
df= df.dropna(subset=['clean_text'])

#Creating the corpus
df.clean_text=df.clean_text.astype(str) #make sure the text column is strings
corpus =df.clean_text.tolist() #the corpus needs to be a list of strings. pandas object, to list turns into the list.



#create vectors. This is an object, it is used to create the vectors below with a method
tfidf=TfidfVectorizer(tokenizer=tokenize, min_df=25, max_df=0.9, sublinear_tf=True, use_idf=True, 
                      strip_accents=None, lowercase=True, stop_words=stop) 
#OJO, necesita una funcion para tokenizer, ahi es tokenize snowball, que a su ves es el stemmer que use la funcion de tokenize
#parameters for sublinear and use_idf based on documentation recomendations
#min_tf=int eliminates any "word" with tf less than int, so it elimiantes the links. 
#max_df and min_df: if the number is a float betwee 0.0 and 1.0, it represents a proportuion. 
#instead, if it is an integer, it is an absolute count
#So, as it is, i am telling it to ignore words that appear in less than 25 documents, or more than 30% of the documents

tfm=tfidf.fit_transform(corpus) #This is the term document matrix, you are using fit to fit the parameters of the model
#and transform to change the information into the sparse matrix:
#Create a document term matrix, Dij, [Dij] is the count of jth term in ith document
#i indexes 1 ... n, corpora size n = 190325
#j indexes 1....d, vocab size de 9871

#I need to add the target to the sparse matrix. I am not sure if I can use the sparse matrix on the deep learning
#I am turning into a df
df_sparsematrix=pd.DataFrame.sparse.from_spmatrix(tfm) #transforms the scipy sparse matrix into a pandas df, still sparse
#df_sparsematrix.sparse.to_dense() #returns it back from sparse matrix//this CANNOT BE DONE GIVEM MEMORY. JUST USE THE OTHER

#oK, LETS assume we can use the sparse matrix, let's do the train-test split
y=df['Analytic_boolean']
X=df_sparsematrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#I am running out of RAM!!!! need to delete variables occupying space.
del corpus
del df
del df_sparsematrix
del ElectoralTweets
del tf
del X
del y
gc.collect()  # Explicitly call garbage collection



#A problem, X_train and X_test appear as dataframes, but I need them as arrays, do it for all
X_train=X_train.to_numpy()
X_test=X_test.to_numpy()
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()

#You HAVE to pad the sequence
maxlen = 30  # maximum length of a sequence - truncate after this
# This pads (or truncates) the sequences so that they are of the maximum length
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)



#recursive neural network
max_features = 20000  # This is used in loading the data, picks the most common (max_features) words
#notice that it should be unnecesary here, the number of features is 9871, less than max
batch_size = 32
rnn_hidden_dim = 5 #five hidden layer in the neural network
word_embedding_dim = 50 #words are embedded in 50 dims
model_rnn = Sequential() #initializes the model
model_rnn.add(Embedding(max_features, word_embedding_dim))  
#this adds the first layer
#This layer takes each integer in the sequence and embeds it in a 50-dimensional vector
model_rnn.add(SimpleRNN(rnn_hidden_dim, 
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=X_train.shape[1:]))
#The above code adds the 5 hidden layers
model_rnn.add(Dense(1, activation='sigmoid')) #this adds the last layer, which is the 
#one that predicts, sigmoid because the target is boolean

model_rnn.summary() #gives you info on the model

#You need this to get the performance metrics for the model
rmsprop = keras.optimizers.RMSprop(learning_rate = .0001) 
model_rnn.compile(loss='binary_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

#This runs the model, epoch is the number of times it goes through the whole data in the 
#backpropagation gradiant imporvement of the parameters
model_rnn.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=10,
          validation_data=(X_test, y_test))

#Finally, get scores
score, acc = model_rnn.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score) #0.1702
print('Test accuracy:', acc) #0.9595


#Long-short term memory network
model_lstm = Sequential()
model_lstm.add(Embedding(max_features, word_embedding_dim)) 
model_lstm.add(LSTM(50, input_shape=X_train.shape[1:])) #not sure what the 50 is, it says units???
#Also, I think I am not adding more than one hidden layer, check later
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')

model_lstm.summary()

#to get accuracy
model_lstm.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#run the model
model_lstm.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=10,
          validation_data=(X_test, y_test))

score, acc = model_lstm.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score) #1.14e-07
print('Test accuracy:', acc) #0.04

#Gated Recurrent Units network
model_GRU = Sequential()
model_GRU.add(Embedding(max_features, word_embedding_dim)) 
model_GRU.add(GRU(50, dropout=0.2)) #in the code, it was 128, not 50, probably improves that
model_GRU.add(Dense(46, activation='softmax')) #this one was also super different, probably better this way 
#also, softmax is a generalized logit to many dimensions
model_GRU.summary()

#to get accuracy
model_GRU.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#run the model, error, ignore it, lets just not do it for the exercise
model_GRU.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=10,
          validation_data=(X_test, y_test))