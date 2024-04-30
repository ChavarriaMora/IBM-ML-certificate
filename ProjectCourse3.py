#This is the code for the graded project for course 3: supervised models, classificarion
#Elias Chavarria-Mora

import os #for changing working direction
import pandas as pd
import numpy as np  
import seaborn as sns #this improves the quality of plots
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler #scaling all data to same 
from sklearn.model_selection import train_test_split #name says what it is
from sklearn.linear_model import LogisticRegression #logit model
from sklearn.linear_model import LogisticRegressionCV #for regularized logit models, both L1 and L2
from sklearn.neighbors import KNeighborsClassifier #KNN model
from sklearn.svm import LinearSVC #basicm SVM model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score

#Change working direction
path = os.getcwd() #gets current working directory
print(path) #prints the current wd, 'C:\Users\elias'
os.chdir('../../Elias/1 Serious/Academia/University of Pittsburgh/1 Dissertation/Data')
#as a refreshes, ../ tells it to go up by one in the path, so with two up, we are back at C, and then we just put the dir

#load the data
ElectoralTweets= './df_FullDiss_V6.csv' 
df=pd.read_csv(ElectoralTweets)
df.head()
len(df) #number of observations
df.info() #gives information on features/variables

#Reduce the number of features to just metadata
#get the names of columns
column_names = df.columns.tolist()
print(column_names)

#I am checking the categoricals, to make sure I don't have anything with 190000 categories or the like
df.Country.value_counts() #4 categories
df.lang.value_counts()#36 categoris
df.source.value_counts() #too many, checko more in detail
df.possibly_sensitive.value_counts() #boolean 
df.user_verified.value_counts() #boolean
df.user_protected.value_counts() #boolean, also, they are all negative so useless
df.user_location.value_counts() #22, although a bunch are repeated

#Because a bunch of location are repeated, I can recode them, with the following coding

df['user_location']=np.where(df['user_location']=='San José', 'San José, Costa Rica', df['user_location'])
df['user_location']=np.where(df['user_location']=='Guatemala, Centro América', 'Guatemala', df['user_location'])
df['user_location']=np.where(df['user_location']=='HONDURAS', 'Honduras', df['user_location'])
df['user_location']=np.where(df['user_location']=='Guatemala, C.A.', 'Guatemala', df['user_location'])
df['user_location']=np.where(df['user_location']=='Honduras, Centro América', 'Honduras', df['user_location'])
df['user_location']=np.where(df['user_location']=='Costa Rica.', 'Costa Rica', df['user_location'])
df['user_location']=np.where(df['user_location']=='Guatemala City, GT', 'Ciudad de Guatemala', df['user_location'])
df['user_location']=np.where(df['user_location']=='Tegucigalpa, Honduras C.A.', 'Tegucigalpa, Honduras', df['user_location'])
df['user_location']=np.where(df['user_location']=='GUATEMALA CITY', 'Ciudad de Guatemala', df['user_location'])
df['user_location']=np.where(df['user_location']=='Costa Rica ', 'Costa Rica', df['user_location'])
df['user_location']=np.where(df['user_location']=='San José de Costa Rica, CR', 'San José, Costa Rica', df['user_location'])
#lets see if it worked
df.user_location.value_counts() #worked great, only 11 locations now

#to see the unique values on a column
df.source.unique() #jesus, 66 categories

#Also should turn election into object
df.Election=df.Election.astype(object)


#now create a df with less features
df_smaller=df.filter (items = ['Country', 'Election', 'lang', 'source', 'possibly_sensitive', 'user_verified', 
                               'user_location', 'retweet_count', 'like_count', 'quote_count', 'user_tweet_count', 
                               'user_list_count', 'user_followers_count', 'type_of_account', 'Analytic', 'Authenticity'])
df_smaller.head() 
df_smaller.info()

#eliminate missings
df_smaller.dropna(inplace=True) #drops all the rows (cases) with missing values from the original dataframe


#pairplot first
#to check if it works, I'm doing a much smaller df, with 100 observations
df_verysmall=df_smaller.head(100)
#df_verysmall=df_smaller.head(10000)
sns.set_context('talk') #this is a particular aesthetic in seaborn, used for presentations
sns.pairplot(df_verysmall, hue='type_of_account');
#this way I can figure out of there is skewnwss

#ok, the smaller one worked so go for it, but it takes forever
sns.pairplot(df_smaller, hue='type_of_account'); #I think it is fine like this, but if it were numeric, you shouldn't include hue


#lol, one for the diss
df_image_diss=df.filter(items=['retweet_count', 'like_count', 'quote_count', 'type_of_account', 'Analytic', 'Authenticity'])
sns.pairplot(df_image_diss, hue='type_of_account');

#heatmap can also be used to identify the strongets correlation
plt.figure(figsize=(18,18))
sns.heatmap(df_smaller.corr(),annot=True,cmap='RdYlGn') #very green is high positive correlation, 
#very red is high negative correlation, obviosly just for numeric vars


#Now, actual feature transformation
#one-hot encode the categoricals, basically creating dummies for each
#First, you need to create a pd.series object that includes all the categoricals
one_hot_encode_cols = df_smaller.dtypes[df_smaller.dtypes == object]  # filtering by string categoricals
one_hot_encode_cols = one_hot_encode_cols.index.tolist()  # list of categorical fields
df_smaller[one_hot_encode_cols].head().T

#and then here you one-hot encode
df_smaller = pd.get_dummies(df_smaller, columns=one_hot_encode_cols, drop_first=True)
df_smaller.describe().T

#there could be skewness in the data
#first,I begin by turning all integers into floats
int_columns = df_smaller.select_dtypes(include=['int']).columns #which columns are integers
df_smaller[int_columns] = df_smaller[int_columns].astype(float) #turn integers to float    
#then, check if there is skewness
# Create a list of float colums to check for skewing
mask = df_smaller.dtypes == float
float_cols = df_smaller.columns[mask]
skew_limit = 0.75 # define a limit above which we will log transform; normal distribution skew==0
skew_vals = df_smaller[float_cols].skew() #So, which ones are skewed?: 
# Showing the skewed columns
skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {}'.format(skew_limit)))
skew_cols

#all the skewed ones are the count ones:rt count, quote count, like count, users followers count.
# Perform the skew transformation:
for col in skew_cols.index.values:
    df_smaller[col] = df_smaller[col].apply(np.log1p)


#standardize all to the same scale, it is needed for some models such as KNN
scaler=StandardScaler() #first, you have to create an instant of the class, which is to say, 
#you have to create an object that is an instance of the model
#then, you need to creat a list of all the numeric variables, such that it can iterate over them
numeric_features=['retweet_count','like_count','quote_count', 'user_tweet_count', 'user_list_count',
                  'user_followers_count', 'Analytic', "Authenticity"] 
#then, iterate using the scales
for column in [numeric_features]:
    df_smaller[column]=scaler.fit_transform(df_smaller[column])




#train/test slip
#examine breakdown by type, if the breakdown is very different, you need to use stratified split
df_smaller.type_of_account_party.value_counts() #le hizo one hot encoding a type of account, sirve igual, 1=party
#y no, no esta distribuido, el split tiene que ser estratificado.

#ok, separate the target variable from the features, aka the dependent from the independent variables 
#it is the simplest one
y, X = df_smaller['type_of_account_party'], df_smaller.drop(columns='type_of_account_party')
# Split the data into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
#ojo, este NO es para cv, ese es con StrattifiedShuffleSplit


#Logit model

#Normal logit regression
lr = LogisticRegression(solver='liblinear')
lr=lr.fit(X_train, y_train)


#Also, might be worth having this code here: logit with regularization; also, best parameters via cross-validation
#L1 regularized logistic regression
lr_l1 = LogisticRegressionCV(Cs=10, cv=4, penalty='l1', solver='liblinear').fit(X_train, y_train)
# L2 regularized logistic regression
#lr_l2 = LogisticRegressionCV(Cs=10, cv=4, penalty='l2', solver='liblinear').fit(X_train, y_train)
y_pred=lr_l1.predict(X_test)

print(classification_report(y_test, y_pred))

# Plot confusion matrix
sns.set_palette(sns.color_palette())
_, ax = plt.subplots(figsize=(12,12))
ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', annot_kws={"size": 40, "weight": "bold"})  
labels = ['Party', 'Candidate']
ax.set_xticklabels(labels, fontsize=25);
ax.set_yticklabels(labels, fontsize=25);
ax.set_ylabel('Ground truth', fontsize=30);
ax.set_xlabel('Prediction', fontsize=30)


#KNN model
#basic model with k=3
knn = KNeighborsClassifier(n_neighbors=3) #start a class of knn; 
knn = knn.fit(X_train, y_train)
y_pred = knn.predict(X_test) #OJO que le puse igual y_pred, no la mejor idea, deberian de tener nombres diferentes los y_prod
# Precision, recall, f-score from the multi-class support function
print(classification_report(y_test, y_pred))
print('Accuracy score: ', round(accuracy_score(y_test, y_pred), 2))
print('F1 Score: ', round(f1_score(y_test, y_pred), 2))
# Plot confusion matrix
sns.set_palette(sns.color_palette())
_, ax = plt.subplots(figsize=(12,12))
ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', annot_kws={"size": 40, "weight": "bold"})  
labels = ['Party', 'Candidate']
ax.set_xticklabels(labels, fontsize=25);
ax.set_yticklabels(labels, fontsize=25);
ax.set_ylabel('Ground truth', fontsize=30);
ax.set_xlabel('Prediction', fontsize=30)

#technically, it would be better to create a function to loop over a bunch of k's, then select the best one 
#via elbow method, as follows
max_k = 40
error_rates = list() # 1-accuracy

for k in range(1, max_k):    
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn = knn.fit(X_train, y_train)    
    y_pred = knn.predict(X_test)
    error = 1-round(accuracy_score(y_test, y_pred), 4)
    error_rates.append((k, error))
error_results = pd.DataFrame(error_rates, columns=['K', 'Error Rate'])
# Plot Accuracy (Error Rate) results
sns.set_context('talk')
sns.set_style('ticks')
plt.figure(dpi=300)
ax = error_results.set_index('K').plot(figsize=(12, 12), linewidth=6)
ax.set(xlabel='K', ylabel='Error Rate')
ax.set_xticks(range(1, max_k, 2))
plt.title('KNN Elbow Curve')
plt.savefig('knn_elbow.png')

#SVM model
LSVC = LinearSVC()
LSVC.fit(X_train, y_train)
y_pred = LSVC.predict(X_test) 

print(classification_report(y_test, y_pred))

# Plot confusion matrix
sns.set_palette(sns.color_palette())
_, ax = plt.subplots(figsize=(12,12))
ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', annot_kws={"size": 40, "weight": "bold"})  
labels = ['Party', 'Candidate']
ax.set_xticklabels(labels, fontsize=25);
ax.set_yticklabels(labels, fontsize=25);
ax.set_ylabel('Ground truth', fontsize=30);
ax.set_xlabel('Prediction', fontsize=30)
#you can get code on how to put all the metrics together in a much prettier way in the jupyter notebook for logistic regression
