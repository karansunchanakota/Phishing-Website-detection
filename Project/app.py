import numpy as np
import pandas as pd
import os
import io
from flask import Flask, render_template, request, abort, send_file, Response
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
from sklearn import preprocessing

__author__ = 'Karan Sunchanakota'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

output = io.BytesIO()
feature=[]
number=[]

@app.route('/')
def index():
    return render_template("upload.html")

@app.route("/upload", methods=['POST'])
def upload():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
    
    df = df.drop(df.columns[[0,3,5]],axis=1)

    df = df.sample(frac=1).reset_index(drop=True)
    
    urls_without_labels = df.drop('label',axis=1)
    labels = df['label']

# Dividing the data in the ratio of 70:30 [train_data:test_data]

    data_train, data_test, labels_train, labels_test = train_test_split(urls_without_labels, labels, test_size=0.30, random_state=110)
    print(len(data_train),len(data_test),len(labels_train),len(labels_test))

# checking the split of labels in train and test data
    
    print(labels_train.value_counts())
#checking the split for labels_test data
    print(labels_test.value_counts())
    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.fit(data_train,labels_train)
    
# Predicting the result for test data
    prediction_label = random_forest_classifier.predict(data_test)

# Creating confusion matrix and checking the accuracy 
    confusionMatrix = confusion_matrix(labels_test,prediction_label)
    print(confusionMatrix)
    accuracy_score(labels_test,prediction_label)
    c = accuracy_score(labels_test,prediction_label) * 100
    print(confusionMatrix)

    custom_random_forest_classifier = RandomForestClassifier(n_estimators=500, max_depth=20, max_leaf_nodes=10000)
    custom_random_forest_classifier.fit(data_train,labels_train)
    custom_classifier_prediction_label = custom_random_forest_classifier.predict(data_test)

#feature_importances_ : array of shape = [n_features] ------ The feature importances (the higher, the more important the feature).

#feature_importances_  -- This method returns the quantified relative importance in the order the features were fed to the algorithm

    importances = custom_random_forest_classifier.feature_importances_

#std = np.std([tree.feature_importances_ for tree in custom_random_forest_classifier.estimators_],axis=0)   #[[[estimators_ :explaination ---  list of DecisionTreeClassifier ----- (The collection of fitted sub-estimators.)]]]


    indices = np.argsort(importances)[::-1] 
    	
    for f in range(data_train.shape[1]):
	    feature.append(data_train.columns[indices[f]])
	    number.append(importances[indices[f]])
	    print(data_train.columns[indices[f]])
	    print(importances[indices[f]])
	    print(f"{f+1} {data_train.columns[indices[f]]}   :  {importances[indices[f]]} \n")

    print("**** The blue bars are the feature importances of the randomforest classifier, along with their inter-trees variability*****")

# Plot the feature importances of the forest   
    plt.rcParams.update({'font.size':'9'})
    plt.figure()
    plt.title("Feature Importance")
    plt.bar(range(data_train.shape[1]), importances[indices],
        color="b", align="center")   
#yerr=std[indices] -- this is another parameter that can be included if std is calculated above
#and also it gives error bar that's the reason we calculate std above. but here we are not making it plot.

    plt.xticks(range(data_train.shape[1]), data_train.columns[indices], rotation = 90)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.4)
    plt.xlim([-1, data_train.shape[1]])

    plt.rcParams['figure.figsize'] = (35,15)  #this will increase the size of the plot
    plt.savefig(output, format='png')

    return render_template('complete.html', content=c, cM=confusionMatrix, name = "Feature Importance", len = len(feature), feature = feature, number = number)

@app.route('/new_plot.png')
def plot_png():
    return Response(output.getvalue(), mimetype="image/png")
#    plt.show()
 
    #     return render_template('complete.html', content=df.to_html())
    #     return df.to_html()
    # return render_template('complete.html')
    # return df.to_html()


if __name__ =="__main__":
    app.run(port=4555, debug=True)

    # target = os.path.join(APP_ROOT, 'images/')
    # df = pd.read_excel(target)
    # return df.tohtml()
#     print(target)

#     if not os.path.isdir(target):
#         os.mkdir(target)

#     for file in request.files.getlist("file"):
#         print(file)
#         filename = file.filename
#         destination = "/".join([target,filename])
#         print(destination)
#         file.save(destination)
#     return render_template("complete.html")

