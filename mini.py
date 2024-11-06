
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

input[type="text"], input[type="password"], input[type="submit"] {
    width: 100%;
    padding: 10px;
    margin-bottom: 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box;
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
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7,min_df=5)

# Fit and transform the training set, transform the test set
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=200,class_weight='balanced')
pac.fit(tfidf_train, y_train)


# Predict on the test set\
y_pred = pac.predict(tfidf_test)

# Calculate and display the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display the confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Display the classification report for precision, recall, and F1-score
print("Classification Report:\n", classification_report(y_test, y_pred))

# Define functions for Streamlit app
def main():
    # Inject CSS
    st.markdown(css, unsafe_allow_html=True)
    
    st.title('Fake News Detection')

    # Navigation bar
    menu = ['Home', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        show_homepage()

    elif choice == 'About':
        show_about()

   

def show_homepage():
    st.header('Enter a news headline to check if it\'s real or fake:')
    news_text = st.text_area('Enter news headline:')
    if st.button('Check'):
        if news_text:
            prediction = predict_news(news_text)
            st.subheader('Prediction:')
            st.write(prediction)
        else:
            st.warning('Please enter a news headline.')

def show_about():
    st.header('ABOUT')
    st.markdown("""
    This web application predicts whether a news headline is real or fake using a machine learning model trained on labeled news articles.
    """)



def predict_news(news_text):
    example_tfidf = tfidf_vectorizer.transform([news_text])
    prediction = pac.predict(example_tfidf)
    return prediction[0]

if __name__ == '__main__':
    main()

