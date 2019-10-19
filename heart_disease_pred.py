# imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#%matplotlib inline


df = pd.read_csv('heart.csv')
print('---------------------------------First 5 Rows--------------------------------------')
print(df.head())

print('-----------------------Dataset Information---------------------------')
print(df.info())


print('-------------------------Health List---------------------------------')
print(df['target'].value_counts())

Num_of_Healthy = df[(df['target'] == 0)].count()[1]
Num_of_Sick = df[(df['target'] == 1)].count()[1]

print()
print('Number of people without Heart Disease:'+ str(Num_of_Healthy))
print('Number of people with Heart Disease:' +str(Num_of_Sick))


#Normalizing and Splitting the data

# spliting data table into data X and class label y
x = df.iloc[:,0:13].values
y = df.iloc[:,13].values


#Normalizing the Data
X_std = StandardScaler().fit_transform(x)
df_Normalized = pd.DataFrame(X_std,index=df.index, columns = df.columns[0:13])

#Adding the Target(Class)
df_Normalized['target'] = df['target']
print()
print(df_Normalized.head())

# spliting the Normalized data table into data X and class label y
X = df_Normalized.iloc[:,0:13]
Y = df_Normalized.iloc[:,13]

#Splitting into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
X, Y, test_size=0.3, random_state=0)

print('---------Train/Test Split Numbers---------\n')
print(X_train.shape, y_train.shape, X_test.shape , y_test.shape)

# calculating the Correlation Matrix
corr = df_Normalized.corr()

# plotting the heatmap
#Heatmap makes it easy to identify which features are most related to the target variable,
# we will plot heatmap of correlated features using the seaborn library.
fig = plt.figure(figsize=(5,4))
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
            linewidths=.75)
plt.show()

#Feature Importance

model = ExtraTreesClassifier()
model.fit(X,Y)

print(model.feature_importances_)

#Visualization
features_importances = pd.Series(model.feature_importances_, index=X.columns)
features_importances.nlargest(12).plot(kind='barh')
plt.show()

#Principal component analysis (PCA) is a technique used to emphasize variation and bring out strong patterns in a dataset.
#we use it first make data easy to explore and visualize.

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [1,0]
colors = ['r',  'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()



#Helper Function

results_test = {}
results_train = {}
list_algos=[]

def predict_date(algo_name,X_train,y_train,X_test,y_test,atype='',verbose=0):
    algo_name.fit(X_train, y_train)
    Y_pred = algo_name.predict(X_test)
    acc_train = round(algo_name.score(X_train, y_train) * 100, 2)
    acc_val = round(algo_name.score(X_test, y_test) * 100, 2)

    results_test[str(algo_name)[0:str(algo_name).find('(')]+'_'+str(atype)] = acc_val
    results_train[str(algo_name)[0:str(algo_name).find('(')]+'_'+str(atype)] = acc_train
    list_algos.append(str(algo_name)[0:str(algo_name).find('(')])
    if verbose ==0:
        print("Train Accuracy: " + str(acc_train))
        print("Test Accuracy: "+ str(acc_val))
    else:
        return Y_pred


#Running Random Forest Algorithm on the Features
random_forest = RandomForestClassifier(n_estimators=50, random_state = 0)
predict_date(random_forest,X_train,y_train,X_test,y_test)

#Feature Importance(subjected to Random Forest algorithm)
feature_importance = random_forest.feature_importances_
feat_importances = pd.Series(random_forest.feature_importances_, index=df.columns[:-1])
feat_importances = feat_importances.nlargest(13)

feature = df.columns.values.tolist()[0:-1]
importance = sorted(random_forest.feature_importances_.tolist())


x_pos = [i for i, _ in enumerate(feature)]

plt.barh(x_pos, importance , color='dodgerblue')
plt.ylabel("feature")
plt.xlabel("importance")
plt.title("feature_importances")

plt.yticks(x_pos, feature)
plt.show()


#Performing Predictions on the Test Data in our Main Method
def main():
    print ("Trained model :: ", random_forest)
    predictions = random_forest.predict(X_test)

    for i in range(0, len(X_test)- 1):

        print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(y_test)[i], predictions[i]))

    print ("Train Accuracy :: ", accuracy_score(y_train, random_forest.predict(X_train)))
    print ("Test Accuracy  :: ", accuracy_score(y_test, predictions))
    print (" Confusion matrix ", confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()

