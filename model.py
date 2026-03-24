import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from preprocess import preprocess_text

# Load dataset
data = pd.read_csv("data/spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Preprocess
data['cleaned'] = data['message'].apply(preprocess_text)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['cleaned'])
y = data['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
lr = LogisticRegression()
nb = MultinomialNB()
svm = LinearSVC()

# Train
lr.fit(X_train, y_train)
nb.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Evaluate
print("Logistic Regression:\n", classification_report(y_test, lr.predict(X_test)))
print("Naive Bayes:\n", classification_report(y_test, nb.predict(X_test)))
print("SVM:\n", classification_report(y_test, svm.predict(X_test)))

# Choose best model (example: SVM)
model = svm

# Export objects
def get_model():
    return model, tfidf