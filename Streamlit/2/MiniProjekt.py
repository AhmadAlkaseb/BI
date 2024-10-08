#!/usr/bin/env python
# coding: utf-8

# # Define business requirement
# #### By Ahmad & Hanni

# **Hypothesis**:  
# *States with more relaxed gun ownership regulations experience a higher frequency of school shootings compared to states with stricter gun control policies.*

# ## Data collection

# *Here I am importing the needed libraries:*

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import graphviz
from sklearn import metrics, tree, model_selection
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from scipy.spatial.distance import cdist


# *Next I am loading the data into a dataframe using pandas library.*

# In[7]:


df = pd.read_csv('Mass shooting data.csv')


# ## Data cleaning

# *Now I am having a look at the dataframe:*

# In[10]:


df


# *Now I want to extend the information from the dataframe:*

# In[12]:


df.info()


# *Since I don't want to use the following columns:*
# 
# *Operations*  
# *Address*  
# *City Or County*  
# *Incident ID*  
# 
# *I will be deleting them since they won't affect my hypothesis.*

# In[14]:


df = df.drop(['Operations', 'Address', 'City Or County', 'Incident ID'], axis=1)


# *Let's see, how our dataframe looks now:*

# In[16]:


df.info()


# *I only want to keep the year in my data column:*

# In[18]:


df['Incident Date'] = pd.to_datetime(df['Incident Date'])
df['year'] = df['Incident Date'].dt.year
df = df.drop('Incident Date', axis=1)


# *Now I want to add for each state their law strictness, scoring from 1 to 3:*

# In[20]:


weapon_law_mapping = {
    'Alabama': 1, 'Alaska': 1, 'Arizona': 1, 'Arkansas': 1, 'California': 3,
    'Colorado': 2, 'Connecticut': 3, 'Delaware': 3, 'District of Columbia': 3,
    'Florida': 2, 'Georgia': 1, 'Idaho': 1, 'Illinois': 3, 'Indiana': 1,
    'Iowa': 2, 'Kansas': 1, 'Kentucky': 1, 'Louisiana': 1, 'Maine': 2,
    'Maryland': 3, 'Massachusetts': 3, 'Michigan': 2, 'Minnesota': 2,
    'Mississippi': 1, 'Missouri': 1, 'Montana': 1, 'Nebraska': 2, 'Nevada': 2,
    'New Jersey': 3, 'New Mexico': 1, 'New York': 3, 'North Carolina': 2,
    'Ohio': 2, 'Oklahoma': 1, 'Oregon': 2, 'Pennsylvania': 2, 'South Carolina': 1,
    'South Dakota': 1, 'Tennessee': 1, 'Texas': 1, 'Utah': 1, 'Virginia': 2,
    'Washington': 2, 'West Virginia': 1, 'Wisconsin': 2, 'Wyoming': 1
}

df['Weapon law'] = df['State'].map(weapon_law_mapping)


# *I can conclude that my data has been cleaned and is ready for exploration and analysis.*

# ## Data Exploration & Analysis

# *To start with I want to just have a look at how the dataframe looks:*

# In[24]:


df


# *Here I am getting the first 5 rows:*

# In[26]:


df.head()


# *Here I am getting 5 random rows:*

# In[28]:


df.sample(5)


# *Here I am getting the last 5 rows:

# In[30]:


df.tail()


# *Here I am getting all the columns names printed out:*

# In[32]:


df.columns


# *Here I am printing how many rows, columns and the size of my data:*

# In[34]:


print('Rows: {} Columns: {} Size: {}'.format(df.shape[0], df.shape[1], df.size))


# *Here I am printing the descriptive statistics information:*

# In[36]:


df.describe()


# *Here I am creating histograms:*

# In[38]:


df.hist(figsize=(22, 22))


# *I can conclude that none of them are normally distributed.*

# *Here I am grouping by states to get the maximum number of killings for each state:*

# In[41]:


df_grouped = df.groupby('State')['# Killed'].max().reset_index()
sns.set_style("whitegrid")
plt.figure(figsize=(16, 8))
sns.barplot(x='State', y='# Killed', data=df_grouped)
plt.xticks(rotation=90)
plt.title('Max Killings by State')
plt.xlabel('State')
plt.ylabel('Max Killed')
plt.show()


# *I can conclude that Nevada has the most killings.*

# *Here I am grouping by states to get the maximum number of injured for each state.*

# In[44]:


df_grouped = df.groupby('State')['# Injured'].max().reset_index()
sns.set_style("whitegrid")
plt.figure(figsize=(16, 8))
sns.barplot(x='State', y='# Injured', data=df_grouped)
plt.xticks(rotation=90)
plt.title('Max injured by State')
plt.xlabel('State')
plt.ylabel('Max injured')
plt.show()


# *We can conclude that Nevada has the most injured.*

# *I now want to display the number of school shootings per state:*

# In[47]:


plt.figure(figsize=(12, 6))
sns.countplot(x='State', data=df, order=df['State'].value_counts().index)
plt.title('State Frequency')
plt.xticks(rotation=90)
plt.show()


# *I can conclude that Illinois has the most shootings.*

# *Here I am creating box and whiskers plot of the killings in each accident + showing the mean:*

# In[50]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='# Killed', data=df)
mean_value = df['# Killed'].mean()
plt.scatter(mean_value, 0, color='red', zorder=10, label='Mean')
plt.title('Box Plot of People Killed')
plt.legend()
plt.show()


# *I can conclude that most of the shootings have 0 or 1 killed, and that anything above 2 killed is an outlier.*

# *Here I am grouping by states to get the number of incidents:*

# In[53]:


df_grouped = df.groupby('State').size().reset_index(name='Total Incidents')
plt.figure(figsize=(10, 6))
sns.scatterplot(x='State', y='Total Incidents', data=df_grouped)
plt.xticks(rotation=90)
plt.title('Total Incidents by State')
plt.xlabel('State')
plt.ylabel('Total Incidents')
plt.show()


# *I can conclude that Illinois has the most shootings.*

# *Here I am grouping by states to get the number of injured:*

# In[56]:


df_grouped = df.groupby('State')['# Injured'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.scatterplot(x='State', y='# Injured', data=df_grouped)
plt.title('Total Injured by State')
plt.xlabel('State')
plt.ylabel('Total Injured')
plt.xticks(rotation=90)
plt.show()


# *I can conclude that Illinois has the most injured.*

# *Here I am grouping by states to get the number of killed:*

# In[59]:


df_grouped = df.groupby('State')['# Killed'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.scatterplot(x='State', y='# Killed', data=df_grouped)
plt.title('Total Killed by State')
plt.xlabel('State')
plt.ylabel('Total Killed')
plt.xticks(rotation=90)
plt.show()


# *I can conclude that Texas has the most killed.*

# *Here I am grouping by state and summing the 'injured' and 'killed' values.*

# In[62]:


df_injured = df.groupby('State')['# Injured'].sum().reset_index()
df_killed = df.groupby('State')['# Killed'].sum().reset_index()
df_grouped = pd.merge(df_injured, df_killed, on='State')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='# Injured', y='# Killed', data=df_grouped)
plt.title('Correlation between total injured and killed by state')
plt.xlabel('Injured')
plt.ylabel('Killed')
plt.show()


# *I can conclude that the more injured the more is killed.*

# *Here I am showing correlation heatmap of the dataframe. But first I will one-hot encode my column state, to be able to see it in my
# heatmap.*

# In[65]:


label_encoder = LabelEncoder()
df['State'] = label_encoder.fit_transform(df['State'])


# In[66]:


corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


# *I can conclude that there is an high correlation between injured and killed, while the rest has no correlation.*

# # Data modellering

# ## Linear Regression (Numeric)

# 1) *First I will shuffle the data:*

# In[71]:


df_shuffled = shuffle(df, random_state=42)


# 2) *Now I creating the dependent and independent variabels:*

# In[73]:


DV = '# Killed' 
X = df_grouped[['# Injured']] 
y = df_grouped[DV]


# 3) *Now I am splitting the data using the following model, which I then I will use to train a new model:*

# In[75]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)


# 4) *I am now creating an LineLinearRegression object:*

# In[77]:


model = LinearRegression()


# 5) *Now I am training my model:*

# In[79]:


model.fit(X_train[['# Injured']], y_train)


# *Here I am printing out the following:*
# 
# * *Interception*
# * *Coefficient*
# * *The model*

# In[81]:


print('Interception: ', model.intercept_)
print('Coefficient: ', model.coef_)
print('The formula: Killed = {0:0.2f} + ({1:0.2f} x injured)'.format(model.intercept_, model.coef_[0]))


# 6) *I now want to try my model:*

# In[83]:


y_predictions = model.predict(X_test[['# Injured']])


# 7) *Now I am comparing my y_predictions with my real y.*

# In[85]:


plt.scatter(y_test, y_predictions)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. Actual Values (r = {0:0.2f})'.format(pearsonr(y_test, y_predictions)[0]))
plt.show()


# *I can now conclude that there is an strong correlation between the predicted and true values.*

# 8) *Here I creating my metrics dataframe:*

# In[88]:


metrics_df = pd.DataFrame({'Metric': ['MAE', 
                                      'MSE', 
                                      'RMSE', 
                                      'R-Squared'],
                          'Value': [metrics.mean_absolute_error(y_test, y_predictions),
                                    metrics.mean_squared_error(y_test, y_predictions),
                                    np.sqrt(metrics.mean_squared_error(y_test, y_predictions)),
                                    metrics.explained_variance_score(y_test, y_predictions)]}).round(3)


# *I am here printing out my metrics dataframe:*

# In[90]:


metrics_df


# *I can conclude that my model is 72,7 % accurate.*

# # Multiple Regression (Nominal)

# 1) *First I will shuffle the data:*

# In[94]:


df_shuffled = shuffle(df, random_state=42)


# 2) *Then I am creating the dependent and independent variabels.*

# In[96]:


DV = '# Killed'
X = df_shuffled[['# Injured', 'State']]  
y = df_shuffled[DV]


# 3) *Now I am splitting my data using the following model:*

# In[98]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


# 4) *Now I am creating an object of LinearRegression.*

# In[100]:


model = LinearRegression()


# 5) *Now I am training my model:*

# In[102]:


model.fit(X_train, y_train)


# *Here I am printing out the following:
# 
# * *Interception*
# * *Coefficient*
# * *The formula*

# In[104]:


print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)
feature_names = X.columns
equation = ' + '.join(f'({coef:0.2f} x {name})' for coef, name in zip(model.coef_, feature_names))
print(f'Killed = {model.intercept_:0.2f} + {equation}')


# 6) *Now I am testing my model:*

# In[106]:


y_predictions = model.predict(X_test)


# 7) *Now I am plotting the correlation of predicted and actual values in a scatterplot:*

# In[108]:


plt.scatter(y_test, y_predictions)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. Actual Values (r = {0:0.2f})'.format(pearsonr(y_test, y_predictions)[0]))
plt.show()


# *I can conclude that there is a insufficient correlation between predicted and true values.*

# 8) *Now I am creating the metrics dataframe:*

# In[111]:


metrics_df = pd.DataFrame({'Metric': ['MAE', 
                                      'MSE', 
                                      'RMSE', 
                                      'R-Squared'],
                          'Value': [metrics.mean_absolute_error(y_test, y_predictions),
                                    metrics.mean_squared_error(y_test, y_predictions),
                                    np.sqrt(metrics.mean_squared_error(y_test, y_predictions)),
                                    metrics.explained_variance_score(y_test, y_predictions)]}).round(3)


# *Printing out the metrics dataframe:*

# In[113]:


metrics_df


# *I can conclude that my model is 2,9 % accurate.*

# # Polynominal Regression

# 1) *Here I shuffling the my data:*

# In[117]:


df_shuffled = shuffle(df_grouped, random_state=42)


# 2) *Then I am creating the dependent and independent variabels:*

# In[119]:


DV = '# Killed'
X = df_shuffled[['# Injured']]  
y = df_shuffled[DV] 


# 3) *Then I am splitting my data:*

# In[121]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# 4) *Now I feature scale my data:*

# In[123]:


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# 5) *Now I fit the Polynomial Regression model.*

# In[125]:


pol_reg = LinearRegression()
pol_reg.fit(X_train_poly, y_train)


# 6) *Now I test my model.*

# In[127]:


y_predict = pol_reg.predict(X_test_poly)


# 7) *Now I plot my model into a scatterplot:*

# In[129]:


def viz_polynomial():
    plt.scatter(X_test, y_test, color='red', label='Actual Data')
    plt.plot(X_test, y_predict, color='blue', label='Polynomial Regression')
    plt.title('Polynomial Regression')
    plt.xlabel('# Injured')
    plt.ylabel('# Killed')
    plt.legend()
    plt.show()


# In[130]:


viz_polynomial()


# *I can conclude that my model looks insufficient to use.*

# 8) *Now I calculate the metrics dataframe:*

# In[133]:


metrics_dict = {
    'MAE': metrics.mean_absolute_error(y_test, y_predict),
    'MSE': metrics.mean_squared_error(y_test, y_predict),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_predict)),
    'R-Squared': metrics.r2_score(y_test, y_predict)
}


# *Printing out the metrics dataframe:*

# In[135]:


metrics_dict


# *I can conclude that my model is 14,40% accurate.*

# # Clustering

# 1) *Here I am creating the variabel needed for the clustering:*

# In[139]:


X = (df[['State', '# Killed', '# Injured', 'year', 'Weapon law']])


# 2) *Taking a look at the variabel:*

# In[141]:


X


# 3) *Calculating the distortions:*

# In[143]:


distortions = []
K = range(2,10)
for k in K:
    model = KMeans(n_clusters=k, n_init=10).fit(X)
    distortions.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]) 
print("Distortion: ", distortions)


# 4) *Here I am showing the elbow diagram to see the optimal number of clusters:*

# In[145]:


plt.title('Elbow Method for Optimal K')
plt.plot(K, distortions, 'bx-')
plt.xlabel('K')
plt.ylabel('Distortion')
plt.show()


# *I can here conclude that the optimal of clusters would be: 6*

# 5) *Here I determine the number of max clusters using the silhouette score method.*

# In[148]:


scores = []
K = range(2,10)
for k in K:
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(X)
    score = metrics.silhouette_score(X, model.labels_, metric='euclidean', sample_size=len(X))
    print("\nNumber of clusters =", k)
    print("Silhouette score =", score)
    scores.append(score)


# *Here again I am showing the elbow diagram by silhouette score method.*

# In[150]:


plt.title('Silhouette Score Method for Discovering the Optimal K')
plt.plot(K, scores, 'bx-')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.show()


# *I can here conclude that the optimal number of clusters would be: 3*

# *Time to create the model:*

# 1) *Variabel to hold our number of clusters.*

# In[154]:


num_clusters = 6


# 2) *Creating the model:*

# In[156]:


kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=20)


# 3) *Training the model with my data:*

# In[158]:


kmeans.fit(X)


# 4) *Printing out the labels:*

# In[160]:


np.set_printoptions(threshold=np.inf)
kmeans.labels_


# 5) *Here I want to review the clusters:*

# In[162]:


visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
visualizer.fit(X)
visualizer.show()


# *I can conclude that they're all equally in size, and above 0,50.*

# ## Validate the model

# 1) *Trying out my model:*

# In[166]:


test1 = pd.DataFrame([[1, 2, 0, 3, 4]], columns=['State', '# Killed', '# Injured', 'year', 'Weapon law'])
test2 = pd.DataFrame([[8, 0, 4, 2021, 3]], columns=['State', '# Killed', '# Injured', 'year', 'Weapon law'])

print(kmeans.predict(test1))
print(kmeans.predict(test2) == 1)


# 2) *I will now divide my data into clusters using my model:*

# In[168]:


df['cluster'] = kmeans.predict(df[['State', '# Killed', '# Injured', 'year', 'Weapon law']])


# *I am now ready to use my dataframe for classification.*

# ## Decision-Tree-Classification

# 1) *Here I am Converting the data into an array.*

# In[172]:


array = df.values


# 2) *Now I will divide the data into dependent and independent values:*

# In[174]:


X, y = array[:,:-1], array[:,-1]


# 3) *Separating my input data into classes based on labels:*

# In[176]:


class0 = np.array(X[y==0])
class1 = np.array(X[y==1])
class2 = np.array(X[y==2])
class3 = np.array(X[y==3])
class4 = np.array(X[y==4])
class5 = np.array(X[y==5])
class6 = np.array(X[y==6])


# 4) *Initiating two variabels for the model:*
# * *set_prop = data left for comparing the model.*
# * *seed = random value to always get the same random numbere, when splitting the data.*

# In[178]:


set_prop = 0.2
seed = 7


# 5) *Now I will split the data:*

# In[180]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=set_prop, random_state=seed)


# 6) *Now I am build my decision tree*

# In[182]:


params = {'max_depth': 5}
classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)
gr_data = tree.export_graphviz(classifier, out_file=None, 
                         feature_names = df.columns[:X.shape[1]], class_names = True,        
                         filled=True, rounded=True, proportion = False, special_characters=True)  
dtree = graphviz.Source(gr_data)


# 7) *Loading my decision tree:*

# In[184]:


dtree.render("heart") 


# 8) *Print out my tree:*

# In[186]:


dtree


# 9) *Testing my model:*

# In[188]:


y_testp = classifier.predict(X_test)


# 10) *Printing out the confusion:*

# In[190]:


confusion = pd.crosstab(y_test,y_testp)
confusion


# In[191]:


print ("Accuracy is ", accuracy_score(y_test,y_testp))


# *I can conclude that this model's accuracy is 100 %.*

# 11) *Print out the classification report:*

# In[194]:


class_names = ['Class0', 'Class1', 'Class2', 'Class3', 'Class4', 'Class5']
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
plt.show()


# # Naive Bayes Classification

# 1) *Converting my dataframe into an array:*

# In[197]:


array = df.values


# 2) *Creating two (sub) arrays from the dataframe:*

# In[199]:


X = array[:,0:4] 
Y = array[:,4]


# 3) *Splitting the data using this model:*

# In[201]:


test_set_size = 0.2
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_set_size, random_state=seed)


# 4) *Building my model:*

# In[203]:


model = GaussianNB()
model.fit(X_train, Y_train)


# 5) *Testing my model:*

# In[205]:


model.score(X_test, Y_test)


# 6) *Testing the model*

# In[207]:


prediction = model.predict(X_test)


# 7) *Getting the amount of rows:*

# In[209]:


prediction.shape[0]


# 8) *Calculating accuracy of the model for my real data compared with my predictions:*

# In[211]:


print(accuracy_score(Y_test, prediction))


# 9) *Printing out the classification report:*

# In[213]:


cmat = confusion_matrix(Y_test, prediction)
print(cmat)
print(classification_report(Y_test, prediction))


# *I can conclude that the model's accuracy is 49%. This makes it worse than the decision tree model.*

# # Data Validation

# *Since my decision tree model has an higher accuracy, I will continue with that one.*

# *Validating my decision tree model:*

# In[218]:


k = [[100, 0, 8, 1999, 199]]
my_prediction = classifier.predict(k)
my_prediction


# # Deployment and optimization

# *Exporting my model:*

# In[221]:


joblib.dump(kmeans, 'kmmodel.pkl')


# *How I would load my model:*

# In[223]:


model = joblib.load('kmmodel.pkl')


# #### Conclusion

# *I can after my analysis conclude that there is no correlation between law strictness and amount of school shootings.*
