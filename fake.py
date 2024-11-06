import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# CSS styles
css = """
<style>
body {
    font-family: Arial, sans-serif;
    background-color: #f0f0f0;
    margin: 0;
    padding: 0;
}

.container {
    max-width: 800px;
    margin: 20px auto;
    padding: 20px;
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

h1 {
    text-align: center;
    color: #333;
}

h2 {
    color: #555;
}

p {
    line-height: 1.6;
}

textarea {
    width: 100%;
    padding: 10px;
    margin-bottom: 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    resize: vertical;
}

input[type="submit"] {
    background-color: #4CAF50;
    color: white;
    cursor: pointer;
}

input[type="submit"]:hover {
    background-color: #45a049;
}

.message {
    color: red;
    text-align: center;
    margin-bottom: 10px;
}

.success {
    color: #4CAF50;
    text-align: center;
    margin-bottom: 10px;
}

.error {
    color: red;
    text-align: center;
    margin-bottom: 10px;
}

.sidebar {
    background-color: #f0f0f0;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

a {
    display: block;
    text-align: center;
    margin-top: 20px;
    text-decoration: none;
    color: #007bff;
}
</style>
"""

# Load the dataset
df = pd.read_csv("/home/pavani/majorproject/news.csv")

# Get the independent features (X) and the dependent target (y)
X = df['text']
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5)

# Fit and transform the training set, transform the test set
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Model initialization
models = {
    "Passive Aggressive Classifier": PassiveAggressiveClassifier(max_iter=100, class_weight='balanced'),
    "Logistic Regression": LogisticRegression(max_iter=200, class_weight='balanced'),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine": LinearSVC(class_weight='balanced', max_iter=500),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', max_depth=20, random_state=42)
}

# Define functions for Streamlit app
def main():
    # Inject CSS
    st.markdown(css, unsafe_allow_html=True)
    
    st.title('Fake News Detection')

    # Navigation bar
    menu = ['Home', 'Model Performance', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        show_homepage()

    elif choice == 'Model Performance':
        show_metrics()

    elif choice == 'About':
        show_about()

def show_homepage():
    st.header('Enter a news headline to check if it\'s real or fake:')
    model_choice = st.sidebar.selectbox("Choose a model:", list(models.keys()))
    
    # Train the selected model
    model = models[model_choice]
    model.fit(tfidf_train, y_train)
    
    # Get user input
    news_text = st.text_area('Enter news headline:')
    if st.button('Check'):
        if news_text:
            prediction = predict_news(news_text, model)
            st.subheader('Prediction:')
            st.write(prediction)
        else:
            st.warning('Please enter a news headline.')

def show_metrics():
    st.subheader("Model Performance on Test Set")
    model_choice = st.sidebar.selectbox("Choose a model for performance metrics:", list(models.keys()))
    
    # Train and evaluate the selected model
    model = models[model_choice]
    model.fit(tfidf_train, y_train)
    y_pred = model.predict(tfidf_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    # Classification Report
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

def show_about():
    st.header('ABOUT')
    st.markdown("""
    This web application predicts whether a news headline is real or fake using a machine learning model trained on labeled news articles.
    """)

def predict_news(news_text, model):
    example_tfidf = tfidf_vectorizer.transform([news_text])
    prediction = model.predict(example_tfidf)
    return "Fake" if prediction[0] == 0 else "Real"

if __name__ == '__main__':
    main()

