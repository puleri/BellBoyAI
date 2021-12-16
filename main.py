import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

from tensorflow.python.framework import ops

with open("intents.json") as file:
    data = json.load(file)

# if you edit the intents file, make sure to delete models checkpoint and pickle
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)


except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# if you edit the intents file, make sure to comment out this try except and the model.load
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print('Start talking with the bot! (type goodbye to exit)')
    while True:
        inp = input("You: ")
        if inp.lower() == "quit" or inp.lower() == "goodbye" or inp.lower() == "bye":
            print('Glad to help!')
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if tag == "towels":
            while True:
                inp = input("How many do you need (enter a number 1-20): ")
                if int(inp) > 20:
                    print("The most I am allowed to send is 20, but I will get 20 sent to you.")
                else:
                    print(f'{inp} it is')
                    inp = input("Which kind of towels-- bath (1), hand (2), or washcloth (3)? \nEnter a number between 1-3: ")
                    if int(inp) == 1:
                        towel = "bath towels"
                        print("Thanks. Finally, which room will we be sending these towels to?")
                    elif int(inp) == 2:
                        towel = "hand towels"
                        print("Thanks. Finally, which room will we be sending the hand towels to?")
                    else:
                        towel = "washcloths"
                        print("Thanks. Finally, which room will we be sending these washcloths to?")

                    inp = input("Your room is a number between 1-5000: ")
                    print(f"Great! We will get your {towel} sent to room {inp} as soon as possible!")
                    print("Is there anything else I can help you with?")
                    break

        # print("resuls is ", results)
    else:
        if results[results_index] > 0.8:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
            print(random.choice(responses))
        else:
            print('I didn\'t quite get that. Please try again or ask me a different question.')

chat()
