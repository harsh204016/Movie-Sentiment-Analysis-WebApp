import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential , Input 
from tensorflow.keras.layers import  Embedding , Bidirectional , LSTM , Dense
from tensorflow.keras.callbacks import EarlyStopping



max_features = 20000  # Only consider the top 20k words
maxlen = 200
vocab_size = 50000

train = pd.read_csv("data/train.tsv.zip",sep='\t')
train.drop(['SentenceId','PhraseId'],axis=1,inplace=True)
stop_words = pd.read_csv("data/stopwords.csv")

def text_cleaning(text):
    forbidden_words = stop_words['text'].to_list()
    if text:

        text = ' '.join(text.split('.'))
        text = re.sub('\/',' ',text)
        text = re.sub(r'\\',' ',text)
        text = re.sub(r'((http)\S+)','',text)
        text = re.sub(r'\s+', ' ', re.sub('[^A-Za-z]', ' ', text.strip().lower())).strip()
        text = re.sub(r'\W+', ' ', text.strip().lower()).strip()
        text = [word for word in text.split() if word not in forbidden_words]
        return text
    return []

def text_encoding(train_copy):
    train_copy['Phrase'] = train_copy['Phrase'].apply(lambda x: ' '.join(text_cleaning(x)))
    phrases = train_copy['Phrase'].tolist()
    encoded_phrases = [one_hot(d, vocab_size) for d in phrases]
    train_copy['Phrase'] = encoded_phrases
    return train_copy

train_copy = train[:1000].copy()
train_copy = text_encoding(train_copy)

xtrain , xtest ,ytrain , ytest = train_test_split(train_copy.Phrase,train_copy.Sentiment,
stratify=train_copy.Sentiment,
test_size=0.3,random_state=42)

train_data = xtrain.reset_index(drop=True)
test_data = xtest.reset_index(drop=True)

print(train_data.shape,test_data.shape)


x_train = pad_sequences(train_data, maxlen=maxlen)
x_test = pad_sequences(test_data, maxlen=maxlen)

print("SUCESSFULLYY COME TO END OF PREPROCESSING")


model = Sequential()
inputs = Input(shape=(None,), dtype="int32")
# Embed each integer in a 128-dimensional vector
model.add(inputs)
model.add(Embedding(50000, 128))
# Add 2 bidirectional LSTMs
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
# Add a classifier
model.add(Dense(5, activation="softmax"))
#model = keras.Model(inputs, outputs)
model.summary()

callback =  keras.callbacks.EarlyStopping(patience=3)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, ytrain, batch_size=32, epochs=5, validation_data=(x_test,ytest ),callbacks=[callback])

model.save("model.h5")

