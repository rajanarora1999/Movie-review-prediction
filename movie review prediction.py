from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#tokenizer to break the sentence
tokenizer=RegexpTokenizer(r'\w+')
#r'\w+' is the notation to pick all words
#a set of stopwords
eng_stopwords=set(stopwords.words('english'))
#stemmer
ps=PorterStemmer()
# a function to clean a single review
def getStemmedReview(reviews):
	#convert the whole sentence to lower
    reviews=reviews.lower()
    #since our document was scrapped form html so it has some break tags it will replace those with space
    reviews=reviews.replace("<br /><br />"," ")
    #tokenize
    tokens=tokenizer.tokenize(reviews)
    #remove stopwords
    new_tokens =[token for token in tokens if token not in eng_stopwords]
    #stemming
    stemmed_tokens=[ps.stem(token) for token in new_tokens]
    #form a single string and return 
    cleaned_review=' '.join(stemmed_tokens)
    return cleaned_review
df=pd.read_csv('Train.csv')
x=np.array(df)
#separate the labels and feature data
y=x[:,1]
x=x[:,0]
df=pd.read_csv('Test.csv')  
x_test=np.array(df)
#clean the training data
for i in range(x.shape[0]):
    x[i]=getStemmedReview(x[i])
x_test=x_test.reshape(-1,)
#clean the testing data
for i in range(x_test.shape[0]):
    x_test[i]=getStemmedReview(x_test[i])
cv=CountVectorizer()
#Vectorization
x_vec=cv.fit_transform(x)
x_test_vec=cv.transform(x_test)
#Multinomial naive Bayes
mnb=MultinomialNB()
#Train the model
mnb.fit(x_vec,y)
#predict the answers
ans=mnb.predict(x_test_vec)
# store all the answers in a file
with open('submit.csv','w') as f:
    f.write('Id,label\n')
    for i in range(len(ans)):
        f.write("{},{}\n".format(i,ans[i]))