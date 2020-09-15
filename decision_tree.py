import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

#reading of the data
my_data = pd.read_csv("drug200.csv", delimiter=",")

#selecting the features and the target variable
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])
y = my_data["Drug"].values
print(y[0:5])

#some features in this dataset are categorical such as Sex or BPw,e can convert these features to numerical values by pandas.get_dummies()
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

#spilting of dataset into train and test set
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

#making instance of the DecisionTreeClassifier
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(drugTree) # it shows the default parameters

#Fitting the data with the training feature matrix X_trainset and training response vector y_trainset
drugTree.fit(X_trainset,y_trainset)

#storing the predicted tree in predTree
predTree = drugTree.predict(X_testset)
print (predTree [0:5])
print (y_testset [0:5])

#the accuracy score
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

#visualization
dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')
