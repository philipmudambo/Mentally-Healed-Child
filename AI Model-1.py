#importing libraries from the virtual environment
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk

from nltk.stem import WordNetLemmatizer

#creating an initializer variable
lemmatizer = WordNetLemmatizer()

#creating a variable for loading the intents.json file
intents = json.loads(open('intents.json').read())

#initiliazing empty lists
words = []
classes = []
documents = []
#initiliazing list of words to be ignored
ignoreLetters = ['?', '!', '.', ',']

#intents lookup & tokenization
#tokenizing each pattern into a list of word
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordlist = nltk.word_tokenize(pattern)
        words.extend(wordlist)
        documents.append((wordlist,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#lemmatizing each word in "words", removing characters in "ignoreLetters"(if present), sorting words in alphabetical order, & removing duplicates
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(classes))

#sorting classes in alphabetical order & removing duplicates
classes = sorted(set(classes))

#pickling sorted words(converting the hierarchy into byte stream) & classes lists then saving them
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

#intializing an empty list
training = []
#creating a list of 0's with length=classes & sorted in variable
outputempty = [0] * len(classes)

#looping each document in "documents"
for document in documents:
    #initializing an empty list
    bag = []
    #tokenizing each word in the document & lemmatizing it
    wordpatterns = documents[0]
    wordpatterns = [lemmatizer.lemmatize(word) for word in str(wordpatterns).lower().split()]

    #creating a bag of words(rep. in the doc.) & setting the value of each word to 1 otherwise 0
    for word in words: 
        bag.append(1) if word in wordpatterns else bag.append(0)

    outputrow = list(outputempty)
    outputrow[classes.index(document[1])] = 1
    training.append(bag + outputrow)

#shuffling the list
random.shuffle(training)
training = np.array(training)

trainx = training[:, :len(words)]
trainy = training[:, len(words):]

#creating a new sequential keras model(linear type)
model = tf.keras.Sequential()

#adding a densly connected neural network layer to the model
#the layer has 128 units, input shape determined by the length of the 1st traing example & with a relu activation 
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainx[0]),), activation='relu'))
#adding a dropout layer to the model which randomly sets 50% of the input to 0 at each update during training to reduce or-fitting
model.add(tf.keras.layers.Dropout(0.5))
#adding another densly connected neural net to the model(64 units & relu activation function)
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
#adding another dropout layer with same configuration as the previous one(but with a softmax activation)
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainy[0]), activation='softmax'))

#creating a stochastic gradient decent optimizer(SGD) with a learning rate of .01 & momentum of .9 & an anywhere stop momentum
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

#compiling the model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#training the model
hist = model.fit(np.array(trainx), np.array(trainy), epochs=200, batch_size=5, verbose=1)
#saving the model
model.save('chatbot_model.h5', hist)
print('EXECUTED!')