import streamlit as st
import joblib
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------
# Load models + encoder
# -------------------------------
@st.cache_resource
def load_models():
    # 1) Classical TF-IDF + LR pipeline
    tfidf_lr_pipeline = joblib.load("tfidf_lr_pipeline.pkl")

    # 2) MultiLabelBinarizer
    with open("multilabel_binarizer.pkl", "rb") as f:
        mlb = pickle.load(f)

    # 3) DistilBERT model + tokenizer
    MODEL_DIR = "DiestielBert"   # folder where you saved with save_pretrained()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    bert_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)

    return tfidf_lr_pipeline, mlb, tokenizer, bert_model, device


tfidf_lr_pipeline, mlb, tokenizer, bert_model, device = load_models()

# -------------------------------
# Inference functions
# -------------------------------
def predict_with_tfidf(question: str, top_k=5):
    scores = tfidf_lr_pipeline.decision_function([question]).ravel()
    top_idx = np.argsort(scores)[::-1][:top_k]
    preds = [mlb.classes_[i] for i in top_idx if scores[i] > 0]
    if not preds:  # fallback if all scores <= 0
        preds = [mlb.classes_[top_idx[0]]]
    return preds


def predict_with_bert(question: str, threshold=0.5, min_labels=1, top_k=5):
    enc = tokenizer(question, return_tensors="pt", truncation=True,
                    padding=True, max_length=384).to(device)
    with torch.no_grad():
        logits = bert_model(**enc).logits.squeeze(0)
        probs = torch.sigmoid(logits).cpu().numpy()

    idxs = np.where(probs >= threshold)[0]
    if len(idxs) < min_labels:
        idxs = [int(probs.argmax())]

    preds = [mlb.classes_[i] for i in idxs]
    top_preds = [mlb.classes_[i] for i in np.argsort(probs)[::-1][:top_k]]

    return preds, top_preds


# -------------------------------
# Streamlit Interface
# -------------------------------
st.title("💡 StackAssist: Smart Q&A System")
st.write("Ask a programming question, and I’ll predict relevant tags using **TF-IDF + LR** and **DistilBERT**.")

user_input = st.text_area("🔎 Your Question:")

if st.button("Get Answer"):
    if user_input.strip() == "":
        st.warning("Please enter a question first.")
    else:
        # TF-IDF predictions
        tfidf_results = predict_with_tfidf(user_input)

        # DistilBERT predictions
        bert_results, bert_topk = predict_with_bert(user_input)

        # Display
        st.subheader("🔹 TF-IDF + Logistic Regression")
        st.write(", ".join(tfidf_results))

        st.subheader("🔹 DistilBERT (Multi-label)")
        st.write("Predicted:", ", ".join(bert_results))
        st.write("Top-5:", ", ".join(bert_topk))
