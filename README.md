# Sentiment Analysis using DistilBERT and Hugging Face Transformers

This project performs multi-class sentiment analysis on the [Emotion Dataset](https://huggingface.co/datasets/emotion), fine-tuning a pre-trained DistilBERT model using Hugging Face's Transformers and Datasets libraries. It covers the complete NLP pipeline from data preprocessing to model training, evaluation, deployment, and real-time inference with visualization.

---

## 📌 Project Overview

- **Task**: Emotion classification (e.g., joy, sadness, anger, fear, love, surprise)
- **Dataset**: [Hugging Face Emotion Dataset](https://huggingface.co/datasets/emotion)
- **Model**: `distilbert-base-uncased` (a lightweight version of BERT)
- **Frameworks**: PyTorch, Hugging Face Transformers, Datasets
- **Deployment**: Model pushed to Hugging Face Model Hub for public access
- **Extras**: Visualizes sentiment scores for predictions

---

## 🚀 Features

- Load and explore emotion-labeled text data
- Convert integer labels to class names
- Visualize class distributions
- Tokenize text using Hugging Face’s `AutoTokenizer`
- Fine-tune DistilBERT using the `Trainer` API
- Evaluate model with accuracy and F1-score
- Deploy trained model to Hugging Face Hub
- Perform real-time sentiment inference on new text inputs
- Visualize prediction probabilities using bar plots

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Matplotlib & Seaborn (for visualizations)
- Scikit-learn (metrics)

---

## 📈 Results

The fine-tuned model achieved good performance on validation data, and supports real-time emotion classification. A visualization of predicted emotion probabilities for a given input enhances interpretability.
