# ⚠️ Toxic Comment Classification using Deep Learning

A complete **Multi-Label Toxic Comment Classification** web application built using:

- TensorFlow / Keras
- Bidirectional LSTM
- Streamlit
- NLTK
- Pandas & NumPy

This project predicts different categories of toxic comments in real time and also supports bulk CSV predictions.

---

# 🚀 Features

- ✅ Real-time toxic comment prediction
- ✅ Multi-label classification
- ✅ Bulk CSV upload prediction
- ✅ Interactive Streamlit dashboard
- ✅ Visualization of prediction probabilities
- ✅ Sample test cases
- ✅ Model insights section
- ✅ Text preprocessing using NLTK
- ✅ Download prediction results as CSV

---

# 🧠 Toxicity Labels

The model predicts the following categories:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

---

# 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NLTK
- Pandas
- NumPy
- Matplotlib
- Joblib

---

# 📂 Project Structure

```bash
├── app.py
├── toxic_comment_model.keras
├── tokenizer.pkl
├── lemmatizer.pkl
├── stopwords.pkl
├── requirements.txt
└── README.md
```

---

# 📦 Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/toxic-comment-classifier.git
```

```bash
cd toxic-comment-classifier
```

---

## 2️⃣ Create Virtual Environment (Recommended)

### Windows

```bash
python -m venv .venv
```

Activate environment:

```bash
.venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

---

# 🧹 Text Preprocessing Pipeline

The preprocessing pipeline includes:

- Lowercasing
- Removing special characters
- Tokenization
- Stopword removal
- Lemmatization
- Sequence padding

---

# 🧠 Model Architecture

The project uses a:

## Bidirectional LSTM Neural Network

with:

- Embedding Layer
- Bidirectional LSTM
- Dense Layers
- Sigmoid Activation
- Multi-label classification output

---

# 📊 Application Pages

## 1️⃣ Single Prediction

Predict toxicity for a single comment in real time.

---

## 2️⃣ Bulk Prediction

Upload a CSV file and generate predictions for multiple comments.

Features:
- CSV upload
- Select comment column
- Download prediction results

---

## 3️⃣ Model Insights

Displays:
- Model architecture
- Vocabulary size
- Sequence length
- Project information

---

## 4️⃣ Sample Test Cases

Includes predefined sample comments for testing the model.

---

# 📈 Example Predictions

| Comment | Toxic |
|---|---|
| "I appreciate your help." | ❌ |
| "You are an idiot." | ✅ |
| "I will kill you." | ✅ |

---

# 📸 Streamlit Dashboard

## Single Prediction Page

- Input custom comments
- View prediction probabilities
- Visualize results using charts

## Bulk Prediction Page

- Upload CSV
- Predict multiple comments
- Download output file

---

# 📋 Requirements

Example dependencies:

```txt
streamlit
tensorflow
pandas
numpy
matplotlib
nltk
joblib
scikit-learn
```

---

# 📥 Required NLTK Downloads

The application automatically downloads:

```python
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

---

# 💻 Example Usage

```python
prediction = predict_comment(
    "You are a terrible person."
)

print(prediction)
```

---

# 🎯 Future Improvements

- Add model accuracy metrics
- Deploy using Streamlit Cloud
- Add attention mechanism
- Improve UI/UX
- Add confusion matrix visualization
- Add threshold customization

---

# 🌐 Deployment

You can deploy this app using:

- Streamlit Community Cloud
- Render
- Hugging Face Spaces
- AWS
- Azure

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit changes
4. Push changes
5. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Developed by **Shaik Abdul Kader**

---

# ⭐ Support

If you found this project useful:

- ⭐ Star the repository
- 🍴 Fork the project
- 📢 Share with others

---
