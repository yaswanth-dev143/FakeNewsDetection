import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from IPython.display import display
import ipywidgets as widgets
from sklearn.utils.class_weight import compute_class_weight
# Load the dataset
df = pd.read_csv("news.csv")  # Update the path to where your data is located

# Get the independent features (X) and the dependent target (y)
X = df['text']
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5, ngram_range=(1,2))

# Fit and transform the training set, transform the test set
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Compute class weights based on your training labels
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
# Model initialization
models = {
    "Passive Aggressive Classifier": PassiveAggressiveClassifier(max_iter=100, class_weight='balanced'),
    "Logistic Regression": LogisticRegression(max_iter=200, class_weight= class_weights_dict),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine": LinearSVC(class_weight=class_weights_dict , max_iter=500),
    "Decision Tree": DecisionTreeClassifier(class_weight=class_weights_dict, max_depth=20, random_state=42)
}
# Function to get predictions
def predict_news(news_text, model):
    example_tfidf = tfidf_vectorizer.transform([news_text])
    prediction = model.predict(example_tfidf)
    return "Real" if prediction[0] == 0 else "Fake"

# Function to display model metrics
def display_model_metrics(model_choice):
    model = models[model_choice]
    model.fit(tfidf_train, y_train)
    y_pred = model.predict(tfidf_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)
    # Interactive widgets
model_dropdown = widgets.Dropdown(options=list(models.keys()), description="Model:")
text_area = widgets.Textarea(description="News Text:")
button = widgets.Button(description="Check News")
output = widgets.Output()

def on_button_clicked(b):
    with output:
        output.clear_output()
        if text_area.value:
            selected_model = models[model_dropdown.value]
            if not hasattr(selected_model, "coef_"):  # Check if the model is fitted
                selected_model.fit(tfidf_train, y_train)  # Fit the model
            result = predict_news(text_area.value, selected_model)
            print(f"The news is predicted as: {result}")
        else:
            print("Please enter a news headline.")

button.on_click(on_button_clicked)

# Display widgets
display(model_dropdown, text_area, button, output)