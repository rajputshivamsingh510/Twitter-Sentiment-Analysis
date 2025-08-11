# 📊 Twitter Sentiment Analysis

A Machine Learning project to classify tweets into **Positive**, **Negative**, or **Neutral** sentiments using **Natural Language Processing (NLP)** techniques and **TF-IDF vectorization**.

---

## 🚀 Features
- **Data Preprocessing**: Cleans tweets by removing URLs, mentions, hashtags, emojis, punctuation, and stopwords.
- **Tokenization & Lemmatization**: Converts words to their base forms.
- **TF-IDF Vectorization**: Converts text into numerical form for modeling.
- **Model Training**: Supports Logistic Regression, Naive Bayes, and Random Forest classifiers.
- **Evaluation**: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
- **Deployment**: Streamlit web app for real-time sentiment prediction.

---

## 🛠️ Tech Stack
- **Language**: Python
- **Libraries**:
  - NLP: `NLTK`, `re`, `string`
  - Machine Learning: `scikit-learn`
  - Deployment: `Streamlit`
  - Model Saving: `joblib`
- **Tools**: Google Colab (training), Local environment (deployment)

---

## 📂 Project Structure
```
📦 twitter-sentiment-analysis
│── data/
│   └── twitter_training.csv         # Dataset
│── app.py                           # Streamlit app
│── sentiment_model.pkl              # Saved trained model
│── tfidf_vectorizer.pkl             # Saved TF-IDF vectorizer
│── README.md                        # Project documentation
│── requirements.txt                 # Dependencies
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

---

## 📊 Model Training Workflow
1. **Data Cleaning** – Remove unwanted characters, symbols, and noise.
2. **Tokenization** – Split text into individual words.
3. **Stopword Removal** – Filter out common but unimportant words.
4. **Lemmatization** – Reduce words to their base form.
5. **TF-IDF Vectorization** – Convert text to numerical features.
6. **Model Training** – Train ML models and evaluate their performance.
7. **Save Model & Vectorizer** – Store them using `joblib` for later use.

---

## 🧪 Example Predictions
| Tweet | Predicted Sentiment |
|-------|--------------------|
| "Absolutely love the new update!" | Positive |
| "The meeting is at 10 AM tomorrow." | Neutral |
| "Really disappointed with the latest patch." | Negative |

---

## 📜 License
This project is licensed under the MIT License.

---

## 👨‍💻 Author
**Your Name** – [GitHub Profile](https://github.com/your-username)
