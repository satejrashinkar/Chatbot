import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.python.tf2 import disable
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from tkinter import *

model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):
    #tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    #Lemmatize each word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence,words,show_details=True):
    #tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    #Bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w==s:
                #assign 1 if current word is in the vocabulary position
                bag[i]=1
                if show_details:
                    print("Found in bag:%s" % w)
    return(np.array(bag))   

def predict_class(sentence,model):
    #filter out predictions below a threshold
    p = bow(sentence,words,show_details=False) 
    res = model.predict(np.array([p]))[0]   
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD ]
    #Sort by probability
    results.sort(key=lambda x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], "probability": str(r[1])})  #The str() function converts the specified value into a string.
    return return_list

def getResponse(ints,intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result     

def chatbot_response(msg):
    ints =  predict_class(msg,model)
    res = getResponse(ints,intents)
    return res

def send():
        msg = EntryBox.get("1.0",'end-1c').strip()  #To get the start and end position of the entered string #Strip is used to remove spaces at the beginning and the end of the string 
        EntryBox.delete("0.0",END)

        if msg != '':
            ChatLog.config(state=NORMAL)    #The button can be clicked by the user
            ChatLog.insert(END,"User:" + msg +'\n\n' )
            ChatLog.config(foreground='#442265', font=("Verdana",12))

            res = chatbot_response(msg)
            ChatLog.insert(END,"System:" + res +'\n\n' )

            ChatLog.config(state=DISABLED) #The button is not clickable
            ChatLog.yview(END)

base = Tk()
base.title("ChatBot")
base.geometry("400x500")
base.resizable(width=FALSE,height=FALSE)

#create chat window
ChatLog = Text(base,bd=0,bg="white",height="20",width="50",font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to chat window
scrollbar = Scrollbar(base, command=ChatLog.yview,cursor="left_ptr")   #Vertical scrollbar and arrow as the mouse pointer
ChatLog['yscrollcommand'] = scrollbar.set

#create button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send ➡️", width="17", height="5",bd=0, bg="#000000", activebackground="#000000",fg='#32de97',command= send )

#Create the box to enter the message
EntryBox = Text(base,bd=0,bg="White",width="29",height="5",font="Arial")

#Place all the components on the screen
scrollbar.place(x=376,y=6,height=386)
ChatLog.place(x=6,y=6,height=386,width=386)
EntryBox.place(x=6,y=401,height=90,width=265)
SendButton.place(x=255, y=401, height=90)

base.mainloop()