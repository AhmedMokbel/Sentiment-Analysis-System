#importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM 
from tensorflow.keras.callbacks import ModelCheckpoint ,ReduceLROnPlateau

Max_Features=30000


def load_data():
   Dataset=pd.read_csv("cleaned_data.csv").head(10000)
   Dataset=Dataset.dropna()
   return Dataset

data=load_data()

def Text_Preprocessing() :
   Dataset=load_data()
   tokenizer = Tokenizer(num_words=Max_Features, split=' ')
   tokenizer.fit_on_texts(Dataset['SentimentText'].tolist())
   X1 = tokenizer.texts_to_sequences(Dataset['SentimentText'].tolist())
   X1 = pad_sequences(X1 )
   Y1=Dataset['Sentiment'].values
   #Y1 = pd.get_dummies(Dataset['Sentiment']).values
   return X1,Y1



   
def split_data() :
    X1,Y1=Text_Preprocessing()
    
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1,
                                                test_size=0.2 ,
                                                random_state = 42)
    return  X1_train, X1_test, Y1_train,Y1_test 


def Build_Rnn_Model():
    embed_dim = 128
    lstm_out = 128
    X1,Y1=Text_Preprocessing()
    model=Sequential()
    #Add Embedding layer
    model.add(Embedding(input_dim=Max_Features,
           output_dim=embed_dim,input_length=X1.shape[1]))

    #Add first layer
    model.add(LSTM(lstm_out,dropout=0.2,recurrent_dropout=0.2 , return_sequences = True) )
    
    #Add second layer
    model.add(LSTM(lstm_out,dropout=0.2,recurrent_dropout=0.2 , return_sequences = True) )
    
    #Add third layer
    model.add(LSTM(lstm_out,dropout=0.2,recurrent_dropout=0.2 , return_sequences = True) )
    
    #Add fourth layer
    model.add(LSTM(lstm_out,dropout=0.2,recurrent_dropout=0.2) )
    
    
    #Add output layer
    model.add(Dense(1,activation='sigmoid'))
    
    #compile the model
    model.compile(optimizer="adam" ,loss="binary_crossentropy" ,metrics=['accuracy'])
    return model

    
    
def checkpoint_model():
  filepath="model.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
  lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
  callbacks_list = [checkpoint,lr_reduce]    
  return callbacks_list
    
    
    

def fit_model():
    X1_train, X1_test, Y1_train, Y1_test=split_data()
    model=Build_Rnn_Model()
    callbacks_list=checkpoint_model()
    model.fit(X1_train,Y1_train,epochs=12,batch_size=32, 
                  validation_data=(X1_test,Y1_test),
                  callbacks=callbacks_list)  
    return model





    





    