#importing the dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle


#reading the data file
df=pd.read_csv(r'C:\Users\Dell\Desktop\New folder (2)\spam.tsv', sep='\t')
print(df.head())

ham = df[df['label'] == 'ham']
spam = df[df['label'] == 'spam']

#creating equal number of test data
ham = ham.sample(spam.shape[0])
data = ham.append(spam, ignore_index=True)

#splitting and training the data
X_train, X_test, y_train, y_test =  train_test_split(data['message'], data['label'], test_size = 0.3, random_state =0, shuffle = True)
svm = Pipeline([("tfidf", TfidfVectorizer()) , ("classifier", SVC(C = 100, gamma='auto'))])
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print(accuracy_score(y_test, y_pred))

test1 = ['Hello, You are learning natural Language Processing']
test2 = ['Hope you are doing good and learning new things !']
test3 = ['Congratulations, You won a lottery ticket worth $1 Million ! To claim call on 446677']

print(svm.predict(test1))
print(svm.predict(test2))
print(svm.predict(test3))

#pickleing the model
pickle_out = open("model_pickle","wb")
pickle.dump(svm,pickle_out)