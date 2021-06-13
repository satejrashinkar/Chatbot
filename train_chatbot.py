import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential     #A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
from keras.layers import Dense,Dropout   #A simple and powerful regularization technique for neural networks and deep learning models is dropout.
from keras.layers import Activation
from keras.optimizers import SGD    #An optimizer is one of the two arguments required for compiling a Keras model
import random   #This module implements pseudo-random number generators for various distributions.

words = []
classes = []
documents = []
ignore_words = ['?','!']
data_file = open('./intents.json').read()
intents = json.loads(data_file)
print(intents)

for intent in intents['intents']:       #check Intents.JSON file
    for pattern in intent ['patterns']:  
        #tokenize each word
        w = nltk.word_tokenize(pattern)     #A tokenizer that divides a string into substrings by splitting on the specified string (defined in subclasses).
        words.extend(w)         #The extend() method adds all the elements of an iterable (list, tuple, string etc.) to the end of the list (append).
        #add documents in the corpus    
        #In linguistics, a corpus (plural corpora) or text corpus is a large and structured set of texts. In corpus linguistics, they are used to do statistical analysis 
        # and hypothesis testing, checking occurrences or validating linguistic rules within a specific language territory.
        documents.append((w , intent['tag']))

        #add classes to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
#sort classes
classes = sorted(list(set(classes)))
# documents  = combination between patterns and intents
print(len(documents), "documents")
#classes = intents
print(len(classes),"classes",classes)
#words = all words , vocabulary
print(len(words),"unique lemmatized words",words)

pickle.dump(words,open('words.pkl','wb'))   #The second argument is 'wb'. The w means that you'll be writing to the file, and b refers to binary mode.
pickle.dump(classes,open('classes.pkl','wb'))

#create training data
training = []
output_empty = [0] * len(classes)       #create a empty array for output
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = []
    #lemmatize each word  - create a base word in attempt to represent the realated word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create a bag of words array with 1 if word found in current position
    for w in words:
        if w in pattern_words:
            bag.append(1) 
        else:
            bag.append(0)    
    # output is a '0' for each tag and '1' for current tag  (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag,output_row])

#shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
#create train and test lists X-patterns and Y-intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),), activation='relu')) #The Sequential model is a linear stack of layers.  
#Applies the rectified linear unit activation function.With default values, this returns the standard ReLU activation: max(x, 0), the element-wise maximum of 0 and the input tensor.
model.add(Dropout(0.5)) #Dropout(rate, noise_shape=None, seed=None, **kwargs) 
#The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.
model.add(Dense(len(train_y[0]),activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='catagorical_crossentropy',optimizer=sgd,metrics= ['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5)
model.save('chatbot_model.h5', hist)

print("Model Created")