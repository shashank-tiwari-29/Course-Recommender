import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def train_sentiment():

    data = pd.read_csv("data/udemy_courses.csv")

    reviews = data["subject"].astype(str)

    labels = [1]*len(reviews)

    tokenizer = Tokenizer(num_words=5000)

    tokenizer.fit_on_texts(reviews)

    seq = tokenizer.texts_to_sequences(reviews)

    padded = pad_sequences(seq,maxlen=50)

    model = Sequential()

    model.add(Embedding(5000,64,input_length=50))
    model.add(Conv1D(64,5,activation="relu"))
    model.add(LSTM(64))
    model.add(Dense(1,activation="sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    model.fit(padded,labels,epochs=5)

    model.save("models/sentiment_model.h5")