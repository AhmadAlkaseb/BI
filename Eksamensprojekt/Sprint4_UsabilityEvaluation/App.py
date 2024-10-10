#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import streamlit as st
import io
from sklearn import metrics, tree, model_selection
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans 
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

file = st.file_uploader("", type="csv")

if file is not None:
    df = pd.read_csv(file)
    st.markdown("# Data Cleaning")

    col1, col2 = st.columns(2)
    with st.expander("Show dataframe before cleaned", expanded=True):
        st.dataframe(df)

    with st.expander("Show dataframe information before cleaned", expanded=True):
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

    df = df.drop(['Operations', 'Address', 'City Or County', 'Incident ID'], axis=1)
    df['Incident Date'] = pd.to_datetime(df['Incident Date'])
    df['year'] = df['Incident Date'].dt.year
    df = df.drop('Incident Date', axis=1)

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

    with st.expander("Show dataframe after cleaned", expanded=True):
        st.dataframe(df)

    with st.expander("Show dataframe information after cleaned", expanded=True):
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

    st.divider()

    st.markdown("# Data Exploration & Analysis")
    col1, col2 = st.columns([1, 1])

    def create_plot(plot_name):
        fig, ax = plt.subplots(figsize=(12, 6))

        if plot_name == "Descriptive statistics":
            return df.describe()

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

        elif plot_name == "Weapon law score by state":
            df_sorted = df.sort_values(by='Weapon law strictness score', ascending=False)
            sns.barplot(x='State', y='Weapon law strictness score', data=df_sorted)
            plt.title('Weapon Law Strictness by State')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        elif plot_name == "Histograms":
            df.hist(ax=ax)

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
        "Weapon law score by state",
        "Descriptive statistics",
        "Histograms",
    ]

    st.markdown("#### Choose diagrams to see")
    col1, col2 = st.columns(2)

    selected_plot_1 = col1.selectbox("", plot_options, key="plot1")
    selected_plot_2 = col2.selectbox("", plot_options, key="plot2")

    with col1:
        result_1 = create_plot(selected_plot_1)
        if isinstance(result_1, pd.DataFrame):
            st.dataframe(result_1)
        else:
            st.pyplot(result_1)

    with col2:
        result_2 = create_plot(selected_plot_2)
        if isinstance(result_2, pd.DataFrame):
            st.dataframe(result_2)
        else:
            st.pyplot(result_2)

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

    def create_download_button(model, model_name):
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)

        st.header("Download the model")

        st.download_button(
            label="Download Model",
            data=buffer,
            file_name=f"{model_name}.pkl",
            mime="application/octet-stream"
        )

    def linear_regression():
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

        y_predictions = model.predict(X_test[['# Injured']])

        coefficient = model.coef_[0]
        intercept = model.intercept_
        formula = f"y = {intercept:.2f} + {coefficient:.2f}x"

        st.latex(formula)

        col1, col2 = st.columns([2, 1])

        with col1:
            plt1 = plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_predictions, color='blue', alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=2)
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title('Predicted vs. Actual Values (r = {0:0.2f})'.format(pearsonr(y_test, y_predictions)[0]))
            equation_text = f'y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x'
            plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
            st.pyplot(plt1)

        with col2:
            metrics_df = pd.DataFrame({
                'Metric': ['MAE', 'MSE', 'RMSE', 'R-Squared'],
                'Value': [
                    metrics.mean_absolute_error(y_test, y_predictions),
                    metrics.mean_squared_error(y_test, y_predictions),
                    np.sqrt(metrics.mean_squared_error(y_test, y_predictions)),
                    metrics.explained_variance_score(y_test, y_predictions)
                ]
            }).round(3)
            st.write("Model Metrics:")
            st.dataframe(metrics_df, hide_index=True)

        st.markdown("## Model Validation")
        st.write("Use the slider below to input a number of injuries and see the predicted number of fatalities.")

        user_input = st.slider("Number of Injuries", 0, 600)

        user_prediction = model.predict([[user_input]])[0]

        st.write(f"Predicted number of killed: {user_prediction:.0f}")

        plot_df = pd.DataFrame({
            'Injuries': X_test['# Injured'],
            'Actual Fatalities': y_test,
            'Predicted Fatalities': y_predictions
        })

        user_point_df = pd.DataFrame({'Injuries': [user_input], 'Predicted Fatalities': [user_prediction]})
        plot_df = pd.concat([plot_df, user_point_df], ignore_index=True)

        create_download_button(model, "linear_model")

    def multiple_regression():

        df_shuffled = shuffle(df, random_state=42)

        DV = '# Killed'
        X = df_shuffled[['# Injured', 'State']]
        y = df_shuffled[DV]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        model = LinearRegression()

        model.fit(pd.get_dummies(X_train), y_train)

        y_predictions = model.predict(pd.get_dummies(X_test))

        feature_names = X.columns
        coefficients = model.coef_
        intercept = model.intercept_

        name_mapping = {
            '# Injured': 'I',
            'State': 'S'
        }

        formula = f"y = {intercept:.4f}"
        for name, coef in zip(feature_names, coefficients):
            short_name = name_mapping.get(name, name)  # If name not in dict, use original name
            formula += f" + ({coef:.4f}{short_name})"

        st.latex(formula)

        col1, col2 = st.columns([2, 1])

        with col1:
            plt2 = plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_predictions)
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title('Predicted vs. Actual Values (r = {0:0.2f})'.format(pearsonr(y_test, y_predictions)[0]))
            st.pyplot(plt2)

        with col2:
            metrics_df = pd.DataFrame({
                'Metric': ['MAE', 'MSE', 'RMSE', 'R-Squared'],
                'Value': [
                    metrics.mean_absolute_error(y_test, y_predictions),
                    metrics.mean_squared_error(y_test, y_predictions),
                    np.sqrt(metrics.mean_squared_error(y_test, y_predictions)),
                    metrics.explained_variance_score(y_test, y_predictions)
                ]
            }).round(3)
            st.write("Model Metrics:")
            st.dataframe(metrics_df, hide_index=True)

        st.markdown("## Model Validation")
        st.markdown("Use the sliders below to input values and see the model's prediction.")

        injured = st.slider("Number of Injured", 0, 600)

        states = X['State'].unique()
        state = st.selectbox("State", states)

        input_data = pd.DataFrame({'# Injured': [injured], 'State': [state]})

        prediction = model.predict(pd.get_dummies(input_data))

        st.write(f"Predicted number of killed: {prediction[0]:.2f}")

        create_download_button(model, "multiple_model")

    def polynomial_regression():

        df_grouped = df.groupby('State').agg({'# Injured': 'sum', '# Killed': 'sum'}).reset_index()
        df_shuffled = shuffle(df_grouped, random_state=42)

        DV = '# Killed'
        X = df_shuffled[['# Injured']]
        y = df_shuffled[DV]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        sc_X = StandardScaler()
        X_train_scaled = sc_X.fit_transform(X_train)
        X_test_scaled = sc_X.transform(X_test)
        poly = PolynomialFeatures(degree=5)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)

        pol_reg = LinearRegression()
        pol_reg.fit(X_train_poly, y_train)

        coefficients = pol_reg.coef_
        intercept = pol_reg.intercept_
        formula = f"y = {intercept:.2f}"
        for i, coef in enumerate(coefficients[1:], 1):
            if coef != 0:
                formula += f" + {coef:.2f}x^{i}"
        st.latex(formula)

        y_predict = pol_reg.predict(X_test_poly)

        plt3 = plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='red', label='Actual Data')
        plt.scatter(X_test, y_predict, color='blue', label='Predicted Data')

        X_test_sorted = np.sort(X_test, axis=0)
        X_test_sorted_scaled = sc_X.transform(X_test_sorted)
        X_test_sorted_poly = poly.transform(X_test_sorted_scaled)
        y_predict_sorted = pol_reg.predict(X_test_sorted_poly)

        col1, col2 = st.columns([2, 1])

        with col1:
            plt3 = plt.figure(figsize=(10, 6))
            plt.scatter(X_test, y_test, color='red', label='Actual Data')
            plt.scatter(X_test, y_predict, color='blue', label='Predicted Data')
            plt.plot(X_test_sorted, y_predict_sorted, color='green', label='Polynomial Regression Line')
            plt.title('Polynomial Regression')
            plt.xlabel('# Injured')
            plt.ylabel('# Killed')
            plt.legend()
            st.pyplot(plt3)

        with col2:
            metrics_dict = {
                'MAE': metrics.mean_absolute_error(y_test, y_predict),
                'MSE': metrics.mean_squared_error(y_test, y_predict),
                'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_predict)),
                'R-Squared': metrics.r2_score(y_test, y_predict)
            }
            st.write("Model Metrics:")
            for metric, value in metrics_dict.items():
                st.write(f"{metric}: {value:.4f}")

        st.markdown("## Model Validation")
        st.markdown("Use the slider below to input values and see the model's prediction.")

        injured = st.slider("Number of Injured", 0, 600)

        input_data = np.array([[injured]])
        input_data_scaled = sc_X.transform(input_data)
        input_data_poly = poly.transform(input_data_scaled)

        prediction = pol_reg.predict(input_data_poly)

        st.write(f"Predicted number of killed: {prediction[0]:.0f}")

        create_download_button(pol_reg, "polynomial_model")

    regression_options = [
        "Linear Regression (Numeric)",
        "Multiple Regression (Nominal)",
        "Polynomial Regression"
    ]

    selected_regression = st.selectbox(
        "Choose a regression type:",
        regression_options
    )

    if selected_regression == "Linear Regression (Numeric)":
        st.subheader("Linear Regression (Numeric)")
        linear_regression()
    elif selected_regression == "Multiple Regression (Nominal)":
        st.subheader("Multiple Regression (Nominal)")
        multiple_regression()
    else:
        st.subheader("Polynomial Regression")
        polynomial_regression()

    st.divider()

    st.markdown("# Clustering")

    X = (df[['State', '# Killed', '# Injured', 'year', 'Weapon law strictness score']])

    distortions = []
    K = range(2, 10)
    for k in K:
        model = KMeans(n_clusters=k, n_init="auto").fit(X)
        distortions.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    scores = []
    for k in K:
        model = KMeans(n_clusters=k, n_init=10)
        model.fit(X)
        score = metrics.silhouette_score(X, model.labels_, metric='euclidean', sample_size=len(X))
        scores.append(score)

    col1, col2 = st.columns(2)

    with col1:
        plt5 = plt.figure(figsize=(10, 6))
        plt.title('Elbow Method for Optimal K')
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('K')
        plt.ylabel('Distortion')
        st.pyplot(plt5)

    with col2:
        plt6 = plt.figure(figsize=(10, 6))
        plt.title('Silhouette Score Method for Optimal K')
        plt.plot(K, scores, 'bx-')
        plt.xlabel('K')
        plt.ylabel('Silhouette Score')
        st.pyplot(plt6)

    np.set_printoptions(threshold=np.inf)

    n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax)

    visualizer.fit(X)

    st.pyplot(fig)

    st.markdown("Validate model")

    df['cluster'] = kmeans.predict(df[['State', '# Killed', '# Injured', 'year', 'Weapon law strictness score']])

    state = st.number_input("State", min_value=0, max_value=50, value=1)
    killed = st.number_input("# Killed", min_value=0, max_value=600)
    injured = st.number_input("# Injured", min_value=0, max_value=600)
    year = st.number_input("Year", min_value=2000, max_value=2030)
    weapon_law_score = st.number_input("Weapon Law Strictness Score", min_value=0, max_value=100)

    user_input_df = pd.DataFrame([[state, killed, injured, year, weapon_law_score]],
                                  columns=['State', '# Killed', '# Injured', 'year', 'Weapon law strictness score'])

    predicted_cluster = kmeans.predict(user_input_df)[0]
    st.write(f"The provided data will be in label: {predicted_cluster}")

    create_download_button(kmeans, "cluster_model")

    st.divider()

    def plot_confusion_matrix(cm):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return fig

    def decision_tree_classification():
        array = df.values
        X, y = array[:,:-1], array[:,-1]
        set_prop = 0.2
        seed = 7
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=set_prop, random_state=seed)
        params = {'max_depth': 5}
        classifier = DecisionTreeClassifier(**params)
        classifier.fit(X_train, y_train)

        gr_data = tree.export_graphviz(classifier, out_file=None,
                                       feature_names = df.columns[:X.shape[1]], class_names = True,
                                       filled=True, rounded=True, proportion = False, special_characters=True)
        st.graphviz_chart(gr_data)

        y_testp = classifier.predict(X_test)

        col1, col2 = st.columns(2)

        with col1:
            cm = confusion_matrix(y_test, y_testp)
            st.pyplot(plot_confusion_matrix(cm))

        with col2:
            st.text("Classification Report:")
            report = classification_report(y_test, y_testp)
            st.text(report)

        accuracy = accuracy_score(y_test, y_testp) * 100
        st.write(f"Accuracy: {accuracy:.0f}%")

        st.write("Try out the model")
        state = st.number_input("State (Numerical)", min_value=0, max_value=100, value=0)
        killed = st.number_input("# Killed", min_value=0, max_value=600, value=0)
        injured = st.number_input("# Injured", min_value=0, max_value=600, value=0)
        year = st.number_input("Year", min_value=2000, max_value=2030, value=2000)
        weapon_law_score = st.number_input("Weapon Law Strictness Score", min_value=0, max_value=100, value=0)

        user_input_df = pd.DataFrame([[state, killed, injured, year, weapon_law_score]],
                                     columns=['State', '# Killed', '# Injured', 'year', 'Weapon law strictness score'])

        my_prediction = classifier.predict(user_input_df)[0]

        st.markdown(f"🎉 Your input has been classified into **Cluster {my_prediction:.0f}**")

        create_download_button(classifier, "decision_tree_model")

    def plot_confusion_matrix(cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Cluster 0', 'Cluster 1', 'Cluster 2'],
                    yticklabels=['Cluster 0', 'Cluster 1', 'Cluster 2'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        return plt.gcf()

    def naive_bayes_classification():
        array = df.values
        X = array[:, 0:5]
        y = array[:, 5]

        test_set_size = 0.2
        seed = 7
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_set_size, random_state=seed)

        modell = GaussianNB()
        modell.fit(X_train, Y_train)

        prediction = modell.predict(X_test)

        col1, col2 = st.columns(2)

        with col1:
            st.write("Confusion Matrix:")
            cm = confusion_matrix(Y_test, prediction)
            st.pyplot(plot_confusion_matrix(cm))

        with col2:
            st.text("Classification Report:")
            report = classification_report(Y_test, prediction)
            st.text(report)

        accuracy = accuracy_score(Y_test, prediction) * 100
        st.write(f"Accuracy: {accuracy:.0f}%")

        st.write("Try out the model")
        state = st.number_input("State (Numerical)", min_value=0, max_value=100, value=0)
        killed = st.number_input("# Killed", min_value=0, max_value=600, value=0)
        injured = st.number_input("# Injured", min_value=0, max_value=600, value=0)
        year = st.number_input("Year", min_value=2000, max_value=2030, value=2000)
        weapon_law_score = st.number_input("Weapon Law Strictness Score", min_value=0.0, max_value=100.0, value=0.0)

        user_input_df = pd.DataFrame([[state, killed, injured, year, weapon_law_score ]],
                                     columns=['State', '# Killed', '# Injured', 'year', 'Weapon Law Strictness Score'])

        my_prediction = modell.predict(user_input_df)[0]

        st.markdown(f"🎉 Your input has been classified into **Cluster {my_prediction:.0f}**")

        create_download_button(modell, "naive_bayes_model")

    st.title("Classification")

    classification_method = st.selectbox(
        "Choose a classification method:",
        ("Decision Tree Classification", "Naive Bayes Classification")
    )

    if classification_method == "Decision Tree Classification":
        decision_tree_classification()
    elif classification_method == "Naive Bayes Classification":
        naive_bayes_classification()