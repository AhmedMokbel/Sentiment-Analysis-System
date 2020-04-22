#importing the libraries
import pandas as pd
import re 
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



"""
def load_data():
   Dataset=pd.read_csv("tweets.csv" ,encoding="latin-1")
   return Dataset
data=load_data()

def count_df():
    Dataset=load_data()
    count_rows=Dataset.iloc[:,0].count()
    return count_rows

def cleaning_data():
 corpus=[]
 Dataset=load_data()
 NumberofRows=count_df()
 for i in range(NumberofRows) :
   tweet=re.sub('[^a-zA-Z]',' ',Dataset['SentimentText'][i])
   tweet=tweet.lower()
   tweet=tweet.split()
   ps=PorterStemmer()
   tweet=[ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
   tweet=' '.join(tweet)
   corpus.append(tweet)
 return corpus

def edit_df():
    Dataset=load_data()
    Dataset['SentimentText']=cleaning_data()
    return Dataset

def save_df():
    Dataset=edit_df()
    Dataset.to_csv("cleaned_data",index=False)
    
save_df()    
"""

def clean_Text(Text):
   corpus=[]
   Text=re.sub('[^a-zA-Z]',' ',Text)
   Text=Text.lower()
   Text=Text.split()
   ps=PorterStemmer()
   Text=[ps.stem(word) for word in Text if not word in set(stopwords.words('english'))]
   Text=' '.join(Text)
   corpus.append(Text)
   return corpus
