import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
#%%
data_train= pd.read_csv("Training.csv")
data_test= pd.read_csv("Testing.csv")
#%%
data=pd.concat([data_train,data_test])
#%%
df = pd.DataFrame(data)
#%%
cols=df.columns
cols = cols[:-1]
#%%
x = df[cols]
y = df['prognosis']
#%%
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#%%
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
scores = cross_val_score(mnb, x_test, y_test, cv=5)
naive_score=scores.mean()
#%%
from sklearn.tree import DecisionTreeClassifier
#%%
print ("DecisionTree")
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
scores = cross_val_score(dt, x_test, y_test, cv=5)
decision_tree_score=scores.mean()
#%%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 50)
knn.fit(x_train,y_train)
scores = cross_val_score(knn, x_test, y_test, cv=5)
knn_score=scores.mean()
#%%
from sklearn.svm import SVC
svc=SVC(kernel="sigmoid")
svc.fit(x_train,y_train)
scores = cross_val_score(svc, x_test, y_test, cv=5)
svc_score=scores.mean()
#%%
#from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(random_state=1)
#rf.fit(x_train,y_train)
#scores = cross_val_score(rf, x_test, y_test, cv=5)
#random_forest_score=scores.mean()
#%%
naive_score=print("Naive Bayes Doğruluk Oranı:",naive_score)
svc_score=print("Support Vektor Machine Doğruluk Oranı:",svc_score)
knn_score=print("KNN Doğruluk Oranı:",knn_score)
decision_tree_score=print("DecisionTree Doğruluk Oranı:",decision_tree_score)
#random_forest_score=print("Random Forest Doğruluk Oranı:",random_forest_score)"""
#%%
def bul():
    kolon1=np.zeros(132).reshape(1,-1)#(1,132) lik 0 satırı oluşturduk
    kolon1=pd.DataFrame(kolon1) #DataFrame tipine dönüştürdük
    print("Ateşiniz Var Mı") #örnek olarak sorumuzu sorduk
    kasıntı_input=int(input("Var ise 1\nYok ise 0'a Basın\n"))
    kolon1.loc(axis=1)[0,1,5,10]=kasıntı_input #burada ise sorudan bağımsız birşekilde sutun numaralarını input değeri ile eşitledik
    #axis=1-->yatay olarak verilen değerler
    output=dt.predict(kolon1) #yeni değerimizi tahmin ettik
    return output
#burada ise girdiğimiz değerler sonucunda bize hastakık değerini tahmin ediyor
#%%
#fonksiyonu çalıştırdık
bul()
