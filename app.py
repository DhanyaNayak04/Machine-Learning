import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler
from flask import Flask, request, jsonify, render_template

# Load dataset
dataset = pd.read_csv('online_shoppers_intention.csv')

# Preprocessing
"""le_month = LabelEncoder()
le_visitor_type = LabelEncoder()

# Encode categorical features
dataset['Month'] = le_month.fit_transform(dataset['Month'])
dataset['VisitorType'] = le_visitor_type.fit_transform(dataset['VisitorType'])"""

# Split into features and target
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
le=LabelEncoder()
for i in [10,15,16]:
    X[:,i]=le.fit_transform(X[:,i])
X[:,-1]=le.fit_transform(X[:,-1])
y=le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)



from imblearn.over_sampling import RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
x_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_scaled, y_train)




# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(x_train_resampled)
X_test_pca = pca.transform(X_test_scaled )

# Train multiple models
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_pca, y_train_resampled)

catboost_model = CatBoostClassifier(iterations=2000, learning_rate=0.1, depth=6, verbose=0, random_seed=42)
catboost_model.fit(X_train_pca, y_train_resampled)

logistic_model = LogisticRegression()
logistic_model.fit(X_train_pca, y_train_resampled)

sgd_model=SGDClassifier(random_state=42)
sgd_model.fit(X_train_pca,y_train_resampled)

extra_trees_model = ExtraTreesClassifier(random_state=42)
extra_trees_model.fit(X_train_pca, y_train_resampled)

import pickle

# Combine all models and preprocessors into a dictionary
all_objects = {
    'RandomForest': rf_model,
    'CatBoost': catboost_model,
    'LogisticRegression': logistic_model,
    'SGDClassifier': sgd_model,
    'ExtraTrees': extra_trees_model,
    'Scaler': sc,
    'PCA': pca
}

# Save the dictionary to a single .pkl file
with open('all_models.pkl', 'wb') as file:
    pickle.dump(all_objects, file)


# Flask app
app = Flask(__name__)

import matplotlib.pyplot as plt
import io
import base64

# Accuracy Calculation
from sklearn.metrics import accuracy_score

# Train multiple models and store their accuracy
model_accuracies = {}

rf_accuracy = rf_model.score(X_test_pca, y_test)
model_accuracies['RandomForest'] = rf_accuracy

catboost_accuracy = catboost_model.score(X_test_pca, y_test)
model_accuracies['CatBoost'] = catboost_accuracy

logistic_accuracy = logistic_model.score(X_test_pca, y_test)
model_accuracies['LogisticRegression'] = logistic_accuracy

extra_trees_accuracy = extra_trees_model.score(X_test_pca, y_test)
model_accuracies['ExtraTrees'] = extra_trees_accuracy

sgd_accuracy=sgd_model.score(X_test_pca, y_test)
model_accuracies['SGDClassifier'] = sgd_accuracy


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    selected_model = request.form.get('model')

    if selected_model == "all":
        # Generate comparison chart
        plt.figure(figsize=(10, 7))  # Larger figure for better visualization
        colors = ['blue', 'green', 'orange', 'purple', 'red']
        bar_positions = range(len(model_accuracies))
        accuracies = list(model_accuracies.values())

        plt.bar(bar_positions, accuracies, color=colors, alpha=0.8)
        plt.title('Model Accuracy Comparison', fontsize=16)
        plt.ylabel('Accuracy Score', fontsize=14)
        plt.xticks(bar_positions, model_accuracies.keys(), fontsize=12, rotation=20)
        # Extend y-axis slightly above 1 for better visibility of labels
        plt.ylim(0, 1.1)  # Extend y-axis slightly above 1 for better visibility of labels
        plt.yticks(np.arange(0, 1.1, 0.1)) 

        # Add accuracy values on top of each bar
        for i, acc in enumerate(accuracies):
            plt.text(
                i, acc + 0.01,  # Position slightly above the bar
                f'{acc:.3f}',  # Format accuracy to 3 decimal places
                ha='center', fontsize=12, color='black'
            )

        # Save chart to a string buffer
        buf = io.BytesIO()
        plt.tight_layout()  # Ensure layout is clean
        plt.savefig(buf, format='png')
        buf.seek(0)
        chart_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        return render_template('index.html', chart_url=chart_url)

    else:
        # Display accuracy of the selected model
        accuracy = model_accuracies.get(selected_model, None)
        return render_template('index.html', selected_model=selected_model, accuracy=accuracy)



@app.route('/predict.html')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Map user inputs from the form
        input_data = {
            'Administrative': float(request.form['Administrative']),
            'Administrative_Duration': float(request.form['Administrative_Duration']),
            'Informational': float(request.form['Informational']),
            'Informational_Duration': float(request.form['Informational_Duration']),
            'ProductRelated': float(request.form['ProductRelated']),
            'ProductRelated_Duration': float(request.form['ProductRelated_Duration']),
            'BounceRates': float(request.form['BounceRates']),
            'ExitRates': float(request.form['ExitRates']),
            'PageValues': float(request.form['PageValues']),
            'SpecialDay': float(request.form['SpecialDay']),
            'Month': le.transform([request.form['Month']])[0] if request.form['Month'] in le.classes_ else 0,
            'OperatingSystems': int(request.form['OperatingSystems']),
            'Browser': int(request.form['Browser']),
            'Region': int(request.form['Region']),
            'TrafficType': int(request.form['TrafficType']),
            'VisitorType': le.transform([request.form['VisitorType']])[0] if request.form['VisitorType'] in le.classes_ else 0,
            'Weekend': int(request.form['Weekend'])  # Binary: 0 or 1
        }

        # Convert input_data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess input
        input_scaled = sc.transform(input_df)
        input_pca = pca.transform(input_scaled)

        # Make predictions
        predictions = {
            'RandomForest': bool(rf_model.predict(input_pca)[0]),
            'CatBoost': bool(catboost_model.predict(input_pca)[0]),
            'LogisticRegression': bool(logistic_model.predict(input_pca)[0]),
            'ExtraTrees': bool(extra_trees_model.predict(input_pca)[0])
        }

        # Ensemble (majority voting)
        final_prediction = max(set(predictions.values()), key=list(predictions.values()).count)

        # Render `predict.html` with the prediction results
        return render_template(
            'predict.html',
            predictions=predictions,
            final_prediction=final_prediction
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True)


