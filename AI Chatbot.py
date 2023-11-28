#importing necessary libraries
import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

#initializing variables
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

#loading the pre-train model
model = load_model('chatbot_model.h5')

#creating a function that tokenizes and lemmatizes
def clean_up_sentence(sentence):
    #tokenizing(converting words to their base forms)
    sentence_words = nltk.word_tokenize(sentence)
    #lemmatizing(splitting text into individual words)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#creating a function that takes input, calls it, creates a bag & returns it
def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#creating a function that takes input, creates a bag, reads the bag to the pre-tarined model & returns list of intents with probability
def predict_class (sentence):
    #creating intelligent variables
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    #setting threshold
    ERROR_THRESHOLD = 0.25
    #creating condtional variable
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    #if that is the case, sort it
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list

#creating a function that will create 2 parameters
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    return result
#"intents_list" contains predicted intents of user input
#"intents_json" contains json file that list of possible intents and responses
#the function then matches predicted intent with the intent in the json & select a random response from the list of responses associated with the intent & returns it as a chatbot response

print("Successful! Bot is running!")

#creating an infinite loop for the main program
while True:
    #user input
    message = input("")
    #prediction of the intent input/message
    ints = predict_class (message)
    #retriving appropriate response from the json file
    res = get_response (ints, intents)
    #printing the chatbot response
    print (res)