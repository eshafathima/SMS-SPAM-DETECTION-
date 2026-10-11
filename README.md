# SMS Spam Classifier

A machine learning project that classifies SMS messages as spam or ham (not spam), with a small Streamlit app for trying it out.

## Dataset
[SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from UCI via Kaggle: 5,572 labeled messages (4,825 ham, 747 spam). Download `spam.csv` and place it in the project folder.

## Approach
- Stratified 80/20 train/test split to handle the class imbalance
- TF-IDF features, fit on the training set only to avoid data leakage
- Two models compared: Multinomial Naive Bayes and Logistic Regression (class_weight="balanced")
- Evaluated with precision, recall, F1 and confusion matrix, not accuracy alone

## Results (test set)
| Model | Spam precision | Spam recall | Spam F1 | Accuracy |
|---|---|---|---|---|
| Naive Bayes | 1.00 | 0.77 | 0.87 | 0.97 |
| Logistic Regression | 0.94 | 0.92 | 0.93 | 0.98 |

Logistic Regression performed better overall. Naive Bayes never flagged a real message as spam but missed about 23% of spam.

## How to run
```bash
pip install pandas scikit-learn streamlit
python spam_classifier.py      # train and evaluate both models
streamlit run app.py           # launch the demo app
```

## Limitations
- Trained on a small, older dataset, so it may struggle with modern spam styles.
- Messages that mix normal and spam-like wording can be misclassified.



