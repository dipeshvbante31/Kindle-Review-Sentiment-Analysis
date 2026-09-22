# Kindle Review Sentiment Analysis

A Natural Language Processing (NLP) project for classifying Kindle product reviews into sentiment classes using different text representation techniques and machine learning models.

## 📌 Project Overview

This project analyzes Kindle customer reviews and builds sentiment classification models using:

* Bag of Words (BOW)
* TF-IDF
* Word2Vec
* Multinomial Naive Bayes
* Logistic Regression

The project also includes text preprocessing, lemmatization, model evaluation, and prediction on new reviews.

## 🎯 Objectives

The main objectives of this project are:

* Clean and preprocess customer reviews
* Convert text into numerical features
* Compare different text vectorization techniques
* Train machine learning models for sentiment classification
* Evaluate model performance
* Generate sentiment predictions for new reviews

## 📂 Dataset

The dataset contains Kindle product reviews with the following relevant columns:

| Column   | Description          |
| -------- | -------------------- |
| `Review` | Customer review text |
| `Label`  | Sentiment label      |

The notebook converts the original labels into binary classes:

```text
0 → Negative
1 → Positive
```

> Make sure the label mapping is consistent with the dataset before interpreting predictions.

## 🔄 NLP Pipeline

The project follows this workflow:

```text
Raw Reviews
     ↓
Data Cleaning
     ↓
Lowercasing
     ↓
Remove HTML / URLs / Special Characters
     ↓
Stopword Removal
     ↓
Lemmatization
     ↓
Train-Test Split
     ↓
Text Vectorization
     ↓
Machine Learning Model
     ↓
Evaluation
     ↓
New Review Prediction
```

## 🧹 Text Preprocessing

The reviews are cleaned before training.

The preprocessing includes:

* Converting text to lowercase
* Removing HTML tags
* Removing URLs
* Removing special characters
* Removing stopwords
* Lemmatization

The project uses NLTK for stopwords and WordNet lemmatization.

## 📊 Train-Test Split

The dataset is divided into training and testing sets using:

```python
train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=42
)
```

Therefore:

* 75% → Training data
* 25% → Testing data

## 🧮 Bag of Words

Bag of Words represents each review using word-frequency features.

The implementation uses:

```python
CountVectorizer(
    max_features=50000,
    min_df=2,
    max_df=0.95
)
```

Sparse matrices are used to keep memory usage manageable for the large text dataset.

## 📈 TF-IDF

TF-IDF represents words based on their importance within the reviews.

The implementation uses:

```python
TfidfVectorizer(
    max_features=50000,
    min_df=2,
    max_df=0.95
)
```

This helps reduce the importance of words that occur very frequently across documents.

## 🤖 Multinomial Naive Bayes

Multinomial Naive Bayes is used for the BOW and TF-IDF representations.

```python
from sklearn.naive_bayes import MultinomialNB

mb = MultinomialNB()
```

The model is trained using the sparse text feature matrices.

## 🧠 Word2Vec

Word2Vec is used to learn numerical representations of words based on their surrounding context.

Configuration:

```python
Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)
```

### Parameters

| Parameter     | Value | Meaning                                    |
| ------------- | ----: | ------------------------------------------ |
| `vector_size` |   100 | Each word is represented using 100 numbers |
| `window`      |     5 | Number of surrounding words considered     |
| `min_count`   |     2 | Ignores very rare words                    |
| `workers`     |     4 | CPU workers used for training              |

## 📄 Document Vector

Word2Vec produces vectors for individual words.

For sentiment classification, the project converts an entire review into a single vector by taking the mean of the vectors of its known words.

```python
def document_vector(review, model):
    words = review.split()

    vectors = [
        model.wv[word]
        for word in words
        if word in model.wv
    ]

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)
```

This produces a 100-dimensional vector for each review.

## 🔬 Logistic Regression

The generated Word2Vec document vectors are used to train Logistic Regression:

```python
from sklearn.linear_model import LogisticRegression

w2v_model = LogisticRegression(max_iter=1000)

w2v_model.fit(xtr_w2v, ytr)
```

The model is then used to predict sentiment on the test reviews.

## 📊 Model Evaluation

The project evaluates the models using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score
* Classification Report

Example:

```python
accuracy_score(yte, y_pred)
```

and:

```python
classification_report(yte, y_pred)
```

## 🔮 Prediction on New Reviews

A trained Word2Vec + Logistic Regression pipeline can be used to classify new reviews.

Example:

```python
new_review = "The product is excellent"

prediction, probability = predict_review(
    new_review,
    w2v,
    w2v_model
)
```

The prediction function:

```python
def predict_review(review, w2v, model):
    vector = document_vector(review, w2v)
    vector = vector.reshape(1, -1)

    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]

    return prediction, probability
```

Example output:

```text
Prediction: 1
Probability: [0.08 0.92]
```

The probability array represents the model's probability for each class.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Gensim
* Jupyter Notebook

## 📁 Project Structure

```text
kindle-review-sentiment-analysis/
│
├── kindle_review_sentiment_analysis.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── dataset/
    └── test.csv
```

> If the dataset is large, it is recommended not to upload the complete dataset to GitHub. Instead, provide instructions for obtaining the dataset.

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/kindle-review-sentiment-analysis.git
```

Move into the project:

```bash
cd kindle-review-sentiment-analysis
```

Install dependencies:

```bash
pip install pandas numpy nltk scikit-learn gensim jupyter
```

Download the required NLTK resources:

```python
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
kindle_review_sentiment_analysis.ipynb
```

## 📦 Requirements

Create a `requirements.txt` containing:

```text
pandas
numpy
nltk
scikit-learn
gensim
jupyter
```

## 💡 Key Learning Outcomes

This project demonstrates practical understanding of:

* NLP preprocessing
* Tokenization
* Stopword removal
* Lemmatization
* Bag of Words
* TF-IDF
* Word embeddings
* Word2Vec
* Document embeddings
* Naive Bayes
* Logistic Regression
* Classification metrics
* Confusion matrices
* Text classification
* Prediction on unseen text

## 🚀 Future Improvements

Possible improvements include:

* Compare more machine learning algorithms
* Hyperparameter tuning
* Use pretrained Word2Vec/GloVe embeddings
* Try CNN/LSTM/Transformer-based models
* Add an interactive Streamlit interface
* Save the trained model using Joblib
* Create an API for real-time sentiment prediction
* Deploy the sentiment classifier online

## 👨‍💻 Author

**Dipesh Bante**

This project was developed as a practical NLP and Machine Learning project to understand text preprocessing, feature extraction, word embeddings, and sentiment classification.
