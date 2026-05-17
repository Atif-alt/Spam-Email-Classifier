import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Select useful columns
data = data[['v1', 'v2']]

# Rename columns
data.columns = ['label', 'message']

# Convert labels into numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Convert text messages into numerical vectors
cv = CountVectorizer()
x = cv.fit_transform(data['message'])

# Labels
y = data['label']

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(x_train, y_train)

# Predict accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# User input
while True:

    msg = input("\nEnter message: ")

    if msg.lower() == "exit":
        print("Program Closed")
        break

    msg_data = cv.transform([msg])

    prediction = model.predict(msg_data)

    if prediction[0] == 1:
        print("Spam Message")
    else:
        print("Not Spam")