from CleaningData import clean_Text
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

Max_Features=30000


def load_data():
   Dataset=pd.read_csv("cleaned_data.csv" ).head(10000)
   Dataset=Dataset.dropna()
   return Dataset

def count_df():
    Dataset=load_data()
    count_rows=Dataset.iloc[:,0].count()
    return count_rows

def Text_Test_preprocessing(Text):
   Dataset=load_data()
   index=count_df()    
   Text=clean_Text(Text)
   Text=" ".join(Text)
   x=Dataset['SentimentText'].tolist()
   x.append(Text)
   tokenizer = Tokenizer(num_words=Max_Features, split=' ')
   tokenizer.fit_on_texts(x)
   X1 = tokenizer.texts_to_sequences(x)
   X1 = pad_sequences(X1)
   x_test=X1[index]
   x_test=np.reshape(x_test,(1,x_test.shape[0]))
   return x_test

def predict_text(Text):
    Text=Text_Test_preprocessing(Text)
    model=load_model("model.hdf5")
    scores=model.predict(Text)
    return scores
    

    