import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle




#Let's import the data from sklearn
from sklearn.datasets import load_wine
wine=load_wine()

#Conver to pandas dataframe
data=pd.DataFrame(data=np.c_[wine['data'],wine['target']],columns=wine['feature_names']+['target'])

#Check data with info function
data.info()


#Let's show a summary of teh dataset where we can see 
# the basic statistic data.
data.describe()




X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33,random_state=42)

X_train.shape


#Create the classifier.

clf=RandomForestClassifier(n_estimators=10, random_state=42)

clf.fit(X_train,y_train.values.ravel())




# Make pickle file of our model
pickle.dump(clf, open("model.pkl", "wb"))







