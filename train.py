import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('/home/minhtranh/works/Sentiment-Analysis/data/IMDB-Dataset-preprocessed.csv')

# Split the dataset into features (text) and target (label)
X = df['review']
y = df['sentiment']

# Encode the labels using LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Transform the text data into vector representation using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Define a function to initialize and train a Decision Tree classifier
def train_decision_tree(X_train, y_train):
    dtc = DecisionTreeClassifier(random_state=42)
    dtc.fit(X_train, y_train)
    return dtc

# Define a function to initialize and train a Random Forest classifier
def train_random_forest(X_train, y_train):
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)
    return rfc

# Train and predict using Decision Tree classifier
dtc = train_decision_tree(X_train_vectorized, y_train)
y_pred_dtc = dtc.predict(X_test_vectorized)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dtc))

# Train and predict using Random Forest classifier
rfc = train_random_forest(X_train_vectorized, y_train)
y_pred_rfc = rfc.predict(X_test_vectorized)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rfc))

# Save the trained models to disk
def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

weight_folder = '/home/minhtranh/works/Sentiment-Analysis/weights'
make_folder(weight_folder)

joblib.dump(dtc, './weights/decision_tree_model.joblib')
joblib.dump(rfc, './weights/random_forest_model.joblib')
joblib.dump(vectorizer, './weights/vectorizer.joblib')
joblib.dump(le, './weights/label_encoder.joblib')

print("Models saved to disk!")


# You can now use the trained models (dtc and rfc) for sentiment analysis on new reviews.
# For example, to predict the sentiment of a new review:
new_review = ["This movie was amazing! I loved every moment of it."]
new_review_vectorized = vectorizer.transform(new_review)
sentiment = dtc.predict(new_review_vectorized)
print("Sentiment:", le.inverse_transform(sentiment))
