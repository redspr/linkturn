# Linkturn — Simple AI-Powered JavaScript URL/URI Extractor

`Linkturn` is a specialized command-line utility for **extracting URLs and URIs from customized JavaScript files**.  
It leverages a **Convolutional Neural Network (CNN) - NLP** model trained on a curated dataset of JavaScript patterns. The model tokenizes and classifies potential matches to minimize **false positives** and **false negatives**.

---

## Key Features

- **AI-Powered Pattern Recognition**  
  Uses a CNN model to learn URL/URI patterns from real-world JavaScript code.

- **False Positive Reduction**  
  Token-based classification ensures that `data:` URIs, base64 blobs, and unrelated strings are excluded.

- **Easy Maintained Training**  
  Easily add new variations of invalid/valid sample on the `lib/samples` folder to be trained. You need morethan 10 variation of false-positive case to be trained well.

- **Lightweight CLI Tool**  
  Minimal dependencies, designed for speed in security testing pipelines.

---

## Installation

```bash
git clone https://github.com/redspr/linkturn.git
cd linkturn
pip install -r requirements.txt
```
---

## Usage

```bash
python linkturn.py <URL>
```

---

## How to Train the Model?

- Create patterns variation of some invalid or valid sample, paste it on corresponding ".txt"
- Run the command
```bash
python trainmodel.py
```