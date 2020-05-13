import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import csv
df = pd.read_csv("breast-cancer-wisconsin.data")
#print(df.head())
df.replace("?", -99999, inplace = True)
df.drop(["id"], axis = 1, inplace=True)

x = np.array(df.drop(["Class"], axis = 1))
y = np.array(df["Class"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

classifier = neighbors.KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)
accuracy = classifier.score(x_test, y_test)
print(accuracy*100)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

"""with open("real_data.csv", "w", newline="") as f:
    thewriter = csv.writer(f)
    thewriter.writerow(["Thickness","Size","Shape","Adhesion","Epithelial Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses"])"""
data = pd.read_csv("real_data.csv", header=None, index_col = False)
#print(data)
now_predict = data.values
#print(type(now_predict))

#now_predict = np.array()
now_predict = now_predict.reshape(len(now_predict),-1)
prediction = classifier.predict(now_predict)
print(prediction)