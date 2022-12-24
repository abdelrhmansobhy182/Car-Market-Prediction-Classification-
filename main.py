import pandas as pd
from sklearn.model_selection import train_test_split
from Pre_processing import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler



import plotly.express as px
#read data
data = pd.read_csv('cars-train.csv')

#data.dropna(how='any',inplace=True)

#mean
data['volume(cm3)'] = data['volume(cm3)'].fillna(data['volume(cm3)'].mean())
data['volume(cm3)'] = data['volume(cm3)'].astype('int')

# get object
object = (data.dtypes == 'object')
object_cols = list(object[object].index);
# get missing data by mode
data[object_cols] = data[object_cols].fillna(data[object_cols].mode().iloc[0])
#data['drive_unit']=data['drive_unit'].interpolate(method ='linear', limit_direction ='forward')
#data.dropna(how='any',inplace=True)

dic = {0: 'cheap', 1: 'expensive',2: 'moderate',3: 'very expensive'}
new = data["car-info"].str.split(",", n=2, expand=True)
data["model"] = new[0]
data["company"] = new[1]
data["year"] = new[2]
# Dropping old Name columns
data.drop(columns=["car-info"], inplace=True)
data['year'] = data['year'].str.replace('\W', '', regex=True)
data['year'] = pd.to_numeric(data['year'], errors='coerce')
data['model'] = data['model'].str.replace('\W', '', regex=True)
data['company'] = data['company'].str.replace('\W', '', regex=True)
data['Price Category'] = data['Price Category'].str.upper()
data = data.reindex(columns = [col for col in data.columns if col != 'Price Category'] + ['Price Category'])
############################################

# get object
object = (data.dtypes == 'object')
object_cols = list(object[object].index);

object = (data.dtypes != 'object')
num_object_cols = list(object[object].index);
feature = ['color' ,'transmission' ,'segment' , 'model','fuel_type','company', 'year','drive_unit']
#feature = ['year','segment','transmission','mileage(kilometers)']
# label encoder,
data[object_cols].head()
# for col in object_cols:
#     #print(f"The distribution of categorical valeus in the {col} is : ")
#     print(data[col].value_counts())


pre = LabelEncoder()
for i in object_cols:
    data[i] = data[i].str.upper()
    data[i]= pre.fit_transform(data[i])


###############################################################################################################
print(object_cols)
print(num_object_cols)
X_ = data.iloc[:,1:12] #Features
Y_ = data['Price Category']
chi_scores = chi2(X_,Y_)
p_values = pd.Series(chi_scores[1],index = X_.columns)
p_values.sort_values(ascending = False , inplace = True)
print(p_values)
p_values.plot.bar()


data.to_csv("all.csv", index=False)

#############################


X=data[feature] #Features
Y=data['Price Category'] #Label
corr = data.corr()


# #Correlation plot
plt.subplots(figsize=(12, 8))
top_corr =corr
sns.heatmap(top_corr, annot=True)
plt.show()

test_data=pd.read_csv('test_all.csv')
x_pred=test_data[feature]
X,minimum,maximum = featureScaling(X,0,1)
x_pred=featureScaling_test(x_pred,0,1,minimum,maximum)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=10)

# scaler = MinMaxScaler()
# print(X_train)
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# x_pred = scaler.transform(x_pred)




knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_predict=knn.predict(x_pred)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
ld_predict = lda.predict(x_pred)

rf = RandomForestClassifier(max_features=10, n_estimators=20)
rf.fit(X_train, y_train)
rf_predict = rf.predict(x_pred)
#############################################################################
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
log_prediction = logreg.predict(x_pred)
##################################################################################
clf = DecisionTreeClassifier().fit(X_train, y_train)
clf_predict=clf.predict(x_pred)
# print("score on test: " + str(lda.score(X_test, y_test)))
# print("score on train: "+ str(lda.score(X_train, y_train)))
################################################################


##################################################################################
submission=pd.DataFrame()
#submission = submission[['car_id']]
submission["Price Category"]=clf_predict
submission.replace({"Price Category": dic} , inplace=True)
submission.to_csv('submission_normalize_after_split.csv')