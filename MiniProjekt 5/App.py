#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import graphviz
import streamlit as st
import io
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

st.markdown("# Can we prevent school shootings?")
st.markdown("#### By Ahmad & Hanni")
st.divider()

st.markdown("# Define business requirement")
st.markdown("**Hypothesis**")
st.markdown("*States with more relaxed gun ownership regulations experience a higher frequency of school shootings compared to states with stricter gun control policies.*")
st.divider()

st.markdown("# Data Collection")
st.markdown("We have our data from Kaggle.")

file = st.file_uploader("", type="csv")


if file is not None:
    df = pd.read_csv(file)
    st.markdown("# Data Cleaning")
    st.markdown("*Now I am having a look at the dataframe:*")
    df

    st.markdown("*Now I want to extend the information from the dataframe:*")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.markdown("*Since I don't want to use the following columns:*")
    st.markdown("*Operations*")
    st.markdown("*Address*")
    st.markdown("*City or County*")
    st.markdown("*Incident ID*")
    st.markdown("*I will be deleting them since they won't affect my hypothesis.*")
    df = df.drop(['Operations', 'Address', 'City Or County', 'Incident ID'], axis=1)

    st.markdown("*Let's see, how our dataframe looks now:*")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)


    st.markdown("*I only want to keep the year in my data column:*")
    df['Incident Date'] = pd.to_datetime(df['Incident Date'])
    df['year'] = df['Incident Date'].dt.year
    df = df.drop('Incident Date', axis=1)


    st.markdown("*Data score collected from: https://everytownresearch.org/rankings/*")
    st.markdown("*Now I want to add for each state their law strictness score.*")
    weapon_law_mapping = {
        'Alabama': 12.50, 'Alaska': 9.00, 'Arizona': 8.50, 'Arkansas': 3.00, 'California': 89.50,
        'Colorado': 63.00, 'Connecticut': 82.50, 'Delaware': 61.50, 'District of Columbia': 69.00,
        'Florida': 27.50, 'Georgia': 5.00, 'Idaho': 5.00, 'Illinois': 83.00, 'Indiana': 16.50,
        'Iowa': 15.50, 'Kansas': 9.50, 'Kentucky': 9.00, 'Louisiana': 20.50, 'Maine': 20.50,
        'Maryland': 75.00, 'Massachusetts': 81.00, 'Michigan': 35.00, 'Minnesota': 53.50,
        'Mississippi': 3.00, 'Missouri': 9.00, 'Montana': 5.00, 'Nebraska': 25.00, 'Nevada': 35.00,
        'New Jersey': 79.00, 'New Mexico': 40.50, 'New York': 83.50, 'North Carolina': 25.00,
        'Ohio': 13.00, 'Oklahoma': 7.50, 'Oregon': 68.00, 'Pennsylvania': 40.00, 'South Carolina': 18.00,
        'South Dakota': 5.00, 'Tennessee': 16.50, 'Texas': 13.50, 'Utah': 12, 'Virginia': 49.00,
        'Washington': 69.00, 'West Virginia': 18.50, 'Wisconsin': 28.00, 'Wyoming': 6.50
    }
    df['Weapon law strictness score'] = df['State'].map(weapon_law_mapping)
    st.markdown("*I can conclude that my data has been cleaned and is ready for exploration and analysis.*")
    st.divider()

    st.markdown("## Data Exploration & Analysis")

    st.markdown("*To start with I want to just have a look at how the dataframe looks:*")
    df

    st.markdown("*Here I am printing the descriptive statistics information:*")
    st.write(df.describe())

    def create_plot(plot_name):
        fig, ax = plt.subplots(figsize=(12, 6))  # Create a figure and axes object

        if plot_name == "Max Killings by State":
            df_grouped = df.groupby('State')['# Killed'].max().reset_index()
            sns.set_style("whitegrid")
            sns.barplot(x='State', y='# Killed', data=df_grouped, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_title('Max Killings by State')
            ax.set_xlabel('State')
            ax.set_ylabel('Max Killed')

        elif plot_name == "Max Injured by State":
            df_grouped = df.groupby('State')['# Injured'].max().reset_index()
            sns.set_style("whitegrid")
            sns.barplot(x='State', y='# Injured', data=df_grouped, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_title('Max Injured by State')
            ax.set_xlabel('State')
            ax.set_ylabel('Max Injured')

        elif plot_name == "Total Killed by State":
            df_grouped = df.groupby('State')['# Killed'].sum().reset_index()
            sns.scatterplot(x='State', y='# Killed', data=df_grouped, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_title('Total Killed by State')
            ax.set_xlabel('State')
            ax.set_ylabel('Total Killed')

        elif plot_name == "Total Injured by State":
            df_grouped = df.groupby('State')['# Injured'].sum().reset_index()
            sns.scatterplot(x='State', y='# Injured', data=df_grouped, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_title('Total Injured by State')
            ax.set_xlabel('State')
            ax.set_ylabel('Total Injured')

        elif plot_name == "Correlation: Injured vs Killed by State":
            df_injured = df.groupby('State')['# Injured'].sum().reset_index()
            df_killed = df.groupby('State')['# Killed'].sum().reset_index()
            df_grouped = pd.merge(df_injured, df_killed, on='State')
            sns.scatterplot(x='# Injured', y='# Killed', data=df_grouped, ax=ax)
            ax.set_title('Correlation between total injured and killed by state')
            ax.set_xlabel('Injured')
            ax.set_ylabel('Killed')

        elif plot_name == "State Frequency":
            sns.countplot(x='State', data=df, order=df['State'].value_counts().index, ax=ax)
            ax.set_title('State Frequency')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        elif plot_name == "Box Plot of People Killed":
            sns.boxplot(x='# Killed', data=df, ax=ax)
            mean_value = df['# Killed'].mean()
            ax.scatter(mean_value, 0, color='red', zorder=10, label='Mean')
            ax.set_title('Box Plot of People Killed')
            ax.legend()

        elif plot_name == "Total Incidents by State":
            df_grouped = df.groupby('State').size().reset_index(name='Total Incidents')
            sns.scatterplot(x='State', y='Total Incidents', data=df_grouped, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_title('Total Incidents by State')
            ax.set_xlabel('State')
            ax.set_ylabel('Total Incidents')

        return fig

    plot_options = [
                    "Max Killings by State",
                    "Max Injured by State",
                    "State Frequency",
                    "Box Plot of People Killed",
                    "Total Incidents by State",
                    "Total Injured by State",
                    "Total Killed by State",
                    "Correlation: Injured vs Killed by State",
        ]

    st.header("Choose a diagram to see")
    selected_plot = st.selectbox("", plot_options)
    st.pyplot(create_plot(selected_plot))

    label_encoder = LabelEncoder()
    df['State'] = label_encoder.fit_transform(df['State'])


    st.header("Correlation heatmap")
    show_heatmap = st.button("Show Correlation Heatmap")
    if show_heatmap:
        corr_matrix = df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        st.pyplot(plt)






    st.divider()



    st.markdown("# Data modellering")

    st.markdown("## Linear Regression (Numeric)")
    df_injured = df.groupby('State')['# Injured'].sum().reset_index()
    df_killed = df.groupby('State')['# Killed'].sum().reset_index()
    df_grouped = pd.merge(df_injured, df_killed, on='State')

    df_grouped = shuffle(df_grouped, random_state=42)

    DV = '# Killed'
    X = df_grouped[['# Injured']]
    y = df_grouped[DV]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = LinearRegression()

    model.fit(X_train[['# Injured']], y_train)
    st.write("Interception: ",model.intercept_)
    st.write("Coefficient: ", model.coef_)
    st.write("The formula: Killed = {0:0.2f} + ({1:0.2f} x injured)".format(model.intercept_, model.coef_[0]))

    y_predictions = model.predict(X_test[['# Injured']])

    plt1 = plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_predictions, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs. Actual Values (r = {0:0.2f})'.format(pearsonr(y_test, y_predictions)[0]))

    # Add text for the equation
    equation_text = f'y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x'
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

    st.pyplot(plt1)

    metrics_df = pd.DataFrame({'Metric': ['MAE',
                                          'MSE',
                                          'RMSE',
                                          'R-Squared'],
                              'Value': [metrics.mean_absolute_error(y_test, y_predictions),
                                        metrics.mean_squared_error(y_test, y_predictions),
                                        np.sqrt(metrics.mean_squared_error(y_test, y_predictions)),
                                        metrics.explained_variance_score(y_test, y_predictions)]}).round(3)

    st.write(metrics_df)

    st.write("Use the slider below to input a number of injuries and see the predicted number of fatalities.")

    # Add this section after your original code
    st.markdown("## Model Validation")
    st.write("Use the slider below to input a number of injuries and see the predicted number of fatalities.")

    # Create a slider for user input
    user_input = st.slider("Number of Injuries", min_value=0, max_value=int(X['# Injured'].max()), value=0)

    # Make prediction based on user input
    user_prediction = model.predict([[user_input]])[0]

    # Display the prediction
    st.write(f"Predicted number of killed: {user_prediction:.0f}")

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Injuries': X_test['# Injured'],
        'Actual Fatalities': y_test,
        'Predicted Fatalities': y_predictions
    })

    # Add user input point using concat instead of append
    user_point_df = pd.DataFrame({'Injuries': [user_input], 'Predicted Fatalities': [user_prediction]})
    plot_df = pd.concat([plot_df, user_point_df], ignore_index=True)



    st.divider()



















    st.markdown("# Multiple Regression (Nominal)")

    st.markdown("1) *First I will shuffle the data:*")

    df_shuffled = shuffle(df, random_state=42)

    st.markdown("2) *Then I am creating the dependent and independent variabels.*")

    DV = '# Killed'
    X = df_shuffled[['# Injured', 'State']]
    y = df_shuffled[DV]

    st.markdown("3) *Now I am splitting my data using the following model:*")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    st.markdown("4) *Now I am creating an object of LinearRegression.*")

    model = LinearRegression()

    st.markdown("5) *Now I am training my model:*")

    model.fit(X_train, y_train)

    st.markdown("Here I am printing out the following:*)")
    st.markdown("* *Interception*")
    st.markdown("* *Coefficient*")
    st.markdown("* *The formula*")

    print('Intercept:', model.intercept_)
    print('Coefficients:', model.coef_)
    feature_names = X.columns
    equation = ' + '.join(f'({coef:0.2f} x {name})' for coef, name in zip(model.coef_, feature_names))
    print(f'Killed = {model.intercept_:0.2f} + {equation}')

    st.markdown("6) *Now I am testing my model:*")

    y_predictions = model.predict(X_test)

    st.markdown("7) *Now I am plotting the correlation of predicted and actual values in a scatterplot:*")

    plt2 = plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs. Actual Values (r = {0:0.2f})'.format(pearsonr(y_test, y_predictions)[0]))
    st.pyplot(plt2)

    st.markdown("*I can conclude that there is a insufficient correlation between predicted and true values.*")

    st.markdown("8) *Now I am creating the metrics dataframe:*")

    metrics_df = pd.DataFrame({'Metric': ['MAE',
                                          'MSE',
                                          'RMSE',
                                          'R-Squared'],
                              'Value': [metrics.mean_absolute_error(y_test, y_predictions),
                                        metrics.mean_squared_error(y_test, y_predictions),
                                        np.sqrt(metrics.mean_squared_error(y_test, y_predictions)),
                                        metrics.explained_variance_score(y_test, y_predictions)]}).round(3)

    st.markdown("*Printing out the metrics dataframe:*")

    metrics_df

    st.markdown("*I can conclude that my model is 2,9 % accurate.*")

    st.markdown("## Model Validation")
    st.markdown("Use the sliders below to input values and see the model's prediction.")

    # Slider for '# Injured'
    injured = st.slider("Number of Injured", int(X['# Injured'].min()), int(X['# Injured'].max()),
                        int(X['# Injured'].mean()))

    # Dropdown for 'State' (assuming 'State' is categorical)
    states = X['State'].unique()
    state = st.selectbox("State", states)

    # Create input data for prediction
    input_data = pd.DataFrame({'# Injured': [injured], 'State': [state]})

    # Make prediction
    prediction = model.predict(input_data)

    st.write(f"Predicted number of killed: {prediction[0]:.2f}")








    st.divider()

    st.markdown("# Polynominal Regression")

    st.markdown("1) *Here I shuffling the my data:*")

    df_shuffled = shuffle(df_grouped, random_state=42)

    st.markdown("2) *Then I am creating the dependent and independent variabels:*")

    DV = '# Killed'
    X = df_shuffled[['# Injured']]
    y = df_shuffled[DV]

    st.markdown("3) *Then I am splitting my data:*")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    st.markdown("4) *Now I feature scale my data:*")

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    poly = PolynomialFeatures(degree=5)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    st.markdown("5) *Now I fit the Polynomial Regression model.*")

    pol_reg = LinearRegression()
    pol_reg.fit(X_train_poly, y_train)

    st.markdown("6) *Now I test my model.*")

    y_predict = pol_reg.predict(X_test_poly)

    st.markdown("7) *Now I plot my model into a scatterplot:*")

    def viz_polynomial():
        plt3 = plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='red', label='Actual Data')
        plt.plot(X_test, y_predict, color='blue', label='Polynomial Regression')
        plt.title('Polynomial Regression')
        plt.xlabel('# Injured')
        plt.ylabel('# Killed')
        plt.legend()
        st.pyplot(plt3)

    viz_polynomial()

    st.markdown("*I can conclude that my model looks insufficient to use.*")

    st.markdown("8) *Now I calculate the metrics dataframe:*")

    metrics_dict = {
        'MAE': metrics.mean_absolute_error(y_test, y_predict),
        'MSE': metrics.mean_squared_error(y_test, y_predict),
        'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_predict)),
        'R-Squared': metrics.r2_score(y_test, y_predict)
    }

    st.markdown("*Printing out the metrics dataframe:*")

    metrics_dict

    st.markdown("*I can conclude that my model is 14,40% accurate.*")

    












    st.divider()

    st.markdown("# Clustering")

    st.markdown("1) *Here I am creating the variabel needed for the clustering:*")

    X = (df[['State', '# Killed', '# Injured', 'year', 'Weapon law strictness score']])

    st.markdown("2) *Taking a look at the variabel:*")

    X

    st.markdown("3) *Calculating the distortions:*")

    distortions = []
    K = range(2,10)
    for k in K:
        model = KMeans(n_clusters=k, n_init="auto").fit(X)
        distortions.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    print("Distortion: ", distortions)

    st.markdown("4) *Here I am showing the elbow diagram to see the optimal number of clusters:*")

    plt.title('Elbow Method for Optimal K')
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('K')
    plt.ylabel('Distortion')
    plt.show()

    st.markdown("*I can here conclude that the optimal of clusters would be: 6*")

    st.markdown("5) *Here I determine the number of max clusters using the silhouette score method.*")

    scores = []
    K = range(2,10)
    for k in K:
        model = KMeans(n_clusters=k, n_init=10)
        model.fit(X)
        score = metrics.silhouette_score(X, model.labels_, metric='euclidean', sample_size=len(X))
        print("\nNumber of clusters =", k)
        print("Silhouette score =", score)
        scores.append(score)

    st.markdown("*Here again I am showing the elbow diagram by silhouette score method.*")

    plt.title('Silhouette Score Method for Discovering the Optimal K')
    plt.plot(K, scores, 'bx-')
    plt.xlabel('K')
    plt.ylabel('Silhouette Score')
    plt.show()

    st.markdown("*I can here conclude that the optimal number of clusters would be: 3*")

    st.markdown("*Time to create the model:*")

    st.markdown("1) *Variabel to hold our number of clusters.*")

    num_clusters = 3

    st.markdown("2) *Creating the model:*")

    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init="auto")

    st.markdown("3) *Training the model with my data:*")

    kmeans.fit(X)

    st.markdown("4) *Printing out the labels:*")

    np.set_printoptions(threshold=np.inf)
    kmeans.labels_

    st.markdown("5) *Here I want to review the clusters:*")

    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    visualizer.fit(X)
    visualizer.show()

    st.markdown("*I can conclude that they're not equally in size, but they're above 0,50.*")
    st.divider()

    st.markdown("## Validate the model")

    st.markdown("1) *Trying out my model:*")

    test1 = pd.DataFrame([[1, 2, 0, 3, 4]], columns=['State', '# Killed', '# Injured', 'year', 'Weapon law strictness score'])
    test2 = pd.DataFrame([[8, 0, 4, 2021, 3]], columns=['State', '# Killed', '# Injured', 'year', 'Weapon law strictness score'])

    print(kmeans.predict(test1))
    print(kmeans.predict(test2) == 1)

    st.markdown("2) *I will now divide my data into clusters using my model:*")

    df['cluster'] = kmeans.predict(df[['State', '# Killed', '# Injured', 'year', 'Weapon law strictness score']])

    st.markdown("*I am now ready to use my dataframe for classification.*")
    st.divider()

    st.markdown("## Decision-Tree-Classification")

    st.markdown("1 *Here I am Converting the data into an array.*")

    array = df.values

    st.markdown("2) *Now I will divide the data into dependent and independent values:*")

    X, y = array[:,:-1], array[:,-1]

    st.markdown("3) *Separating my input data into classes based on labels:*")

    class0 = np.array(X[y==0])
    class1 = np.array(X[y==1])
    class2 = np.array(X[y==2])
    class3 = np.array(X[y==3])
    class4 = np.array(X[y==4])
    class5 = np.array(X[y==5])
    class6 = np.array(X[y==6])

    st.markdown("4) *Initiating two variabels for the model:*")
    st.markdown("* *set_prop = data left for comparing the model.*")
    st.markdown("* *seed = random value to always get the same random numbere, when splitting the data.*")

    set_prop = 0.2
    seed = 7

    st.markdown("5) *Now I will split the data:*")

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=set_prop, random_state=seed)

    st.markdown("6) *Now I am build my decision tree*")

    params = {'max_depth': 5}
    classifier = DecisionTreeClassifier(**params)
    classifier.fit(X_train, y_train)
    gr_data = tree.export_graphviz(classifier, out_file=None,
                             feature_names = df.columns[:X.shape[1]], class_names = True,
                             filled=True, rounded=True, proportion = False, special_characters=True)
    dtree = graphviz.Source(gr_data)

    st.markdown("7) *Loading my decision tree:*")

    dtree.render("heart")

    st.markdown("8) *Print out my tree:*")

    dtree

    st.markdown("9) *Testing my model:*")

    y_testp = classifier.predict(X_test)

    st.markdown("10) *Printing out the confusion:*")

    confusion = pd.crosstab(y_test,y_testp)
    confusion

    st.markdown(confusion)

    print ("Accuracy is ", accuracy_score(y_test,y_testp))

    st.markdown("*I can conclude that this model's accuracy is 100 %.*")

    st.markdown("11) *Print out the classification report:*")

    class_names = ['Class0', 'Class1', 'Class2']
    print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
    plt.show()
    st.divider()

    st.markdown("# Naive Bayes Classification")

    st.markdown("1) *Converting my dataframe into an array:*")

    array = df.values

    st.markdown("2) *Creating two (sub) arrays from the dataframe:*")

    X = array[:,0:4]
    Y = array[:, 4]

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Y_binned = pd.cut(Y, bins=bins, labels=labels)

    st.markdown("3) *Splitting the data using this model:*")

    test_set_size = 0.2
    seed = 7
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y_binned, test_size=test_set_size, random_state=seed)

    st.markdown("4) *Building my model:*")

    model = GaussianNB()
    model.fit(X_train, Y_train)

    st.markdown("5) *Testing my model:*")

    model.score(X_test, Y_test)

    st.markdown("6) *Testing the model*")

    prediction = model.predict(X_test)

    st.markdown("7) *Getting the amount of rows:*")

    st.markdown(prediction.shape[0])

    st.markdown("8) *Calculating accuracy of the model for my real data compared with my predictions:*")

    print(accuracy_score(Y_test, prediction))

    st.markdown("9) *Printing out the classification report:*")

    cmat = confusion_matrix(Y_test, prediction)
    print(cmat)
    print(classification_report(Y_test, prediction))

    st.markdown("*I can conclude that the model's accuracy is 42%. This makes it worse than the decision tree model.*")
    st.divider()

    st.markdown("# Data Validation")

    st.markdown("*Since my decision tree model has an higher accuracy, I will continue with that one.*")

    st.markdown("*Validating my decision tree model:*")
    k = [[1, 0, 8, 2023, 3]]
    my_prediction = classifier.predict(k)
    my_prediction
    st.divider()

    st.markdown("# Deployment and optimization")
    st.markdown("*Exporting my model:*")
    joblib.dump(kmeans, 'kmmodel.pkl')

    st.markdown("*Loading my model:*")
    model = joblib.load('kmmodel.pkl')
    st.divider()

    st.markdown("#### Conclusion")
    st.markdown("*I can after my analysis conclude that there is no correlation between law strictness and amount of school shootings.*")