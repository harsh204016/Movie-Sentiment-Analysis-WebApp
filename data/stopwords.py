import nltk
import pandas as pd
nltk.download('stopwords')
from nltk.corpus import stopwords


stopword = pd.DataFrame()
stopword['text'] = stopwords.words('english')
stopword.to_csv("stopwords.csv")
print(stopword.head())

print(len(forbidden_words),len(li))