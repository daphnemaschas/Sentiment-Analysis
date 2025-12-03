# Sentiment Analysis on Tweets

This project aims to perform sentiment analysis on airline tweets using two different approaches: a **Recurrent Neural Network (RNN)** and a **BERT-based transformer model**. The goal is to classify tweets into three sentiment categories: **negative**, **neutral**, and **positive**.

---

## Project Structure

```

Sentiment-Analysis/
│
├── data/
│   └── tweets.csv                 # Raw dataset of airline tweets
│
├── notebooks/
│   ├── rnn_sentiment.ipynb        # Notebook for RNN approach
│   ├── bert_sentiment.ipynb       # Notebook for BERT fine-tuning
│   └── results/                   # Generated during BERT training (ignored in Git)
│        ├── config.json
│        ├── pytorch_model.bin / model.safetensors
│        ├── optimizer.pt
│        └── ...
│
├── src/
│   └── rnn_model.py                   # RNN model definition and training code
│
├── .gitignore                     # Ignore large generated files and python related files
├── requirements.txt               # Requirement file for dependencies
└── README.md                      # This file

````

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd Sentiment-Analysis
````

2. Create a Python environment and install dependencies:

```bash
conda create -n sentiment-env python=3.10
conda activate sentiment-env
pip install -r requirements.txt
```

> `requirements.txt` should include packages like `torch`, `transformers`, `nltk`, `pandas`, `scikit-learn`, etc.

## Usage

### RNN Notebook

* File: `notebooks/rnn_model.ipynb`
* Implements a **simple RNN** for sentiment classification.
* Steps included:

  * Text preprocessing: remove mentions, hashtags, punctuation, stopwords.
  * Stemming and tokenization.
  * Vocabulary creation and sequence encoding.
  * Training, validation, and evaluation using PyTorch.

### BERT Notebook

* File: `notebooks/bert_model.ipynb`
* Implements **BERT fine-tuning** using the Hugging Face Transformers library.
* Steps included:

  * Tokenization using `BertTokenizer`.
  * Dataset preparation (`Dataset.from_dict` or PyTorch tensors).
  * Model fine-tuning using `Trainer`.
  * Evaluation with multiple metrics (accuracy, precision, recall, F1).

---

## Notes

* Large model files and training artifacts (in `results/`) are **ignored by Git**. Use `.gitignore` to prevent pushing them.
* The project is structured to allow extension or switching models easily.

---

## References

* [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)