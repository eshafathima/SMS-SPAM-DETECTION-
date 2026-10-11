import pandas as pd

df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]

print(df.shape)
print(df["label"].value_counts())
print(df.head())
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Labels to numbers
df["y"] = df["label"].map({"ham": 0, "spam": 1})

# Split first, so the test set stays unseen
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["y"],
    test_size=0.2, random_state=42, stratify=df["y"]
)

# TF-IDF: fit on train only, then transform both
tfidf = TfidfVectorizer(lowercase=True, stop_words="english")
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

print(X_train_vec.shape, X_test_vec.shape)
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Model 1: Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
nb_pred = nb.predict(X_test_vec)

print("\n=== Naive Bayes ===")
print(classification_report(y_test, nb_pred, target_names=["ham", "spam"]))
print(confusion_matrix(y_test, nb_pred))

# Model 2: Logistic Regression (class_weight handles the imbalance)
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train_vec, y_train)
lr_pred = lr.predict(X_test_vec)

print("\n=== Logistic Regression ===")
print(classification_report(y_test, lr_pred, target_names=["ham", "spam"]))
print(confusion_matrix(y_test, lr_pred))
samples = [
    "Congratulations! You won a free iPhone. Click here to claim now",
    "Hey, are we meeting at 5 today?",
    "URGENT: Your account is suspended. Call 09061234567 immediately",
]

sample_vec = tfidf.transform(samples)
for text, pred in zip(samples, lr.predict(sample_vec)):
    print("SPAM" if pred == 1 else "HAM", "->", text)