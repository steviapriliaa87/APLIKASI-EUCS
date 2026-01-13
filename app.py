import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Comment Sentiment & EUCS Dashboard", layout="wide")

DIMS = ["Content", "Accuracy", "Format", "Ease_of_Use", "Timeliness"]
SENTIMENT_MAP = {"negative": 1, "neutral": 2, "positive": 3}
BASE_DIR = Path(__file__).resolve().parent

# =========================
# LOAD MODELS (CACHE)
# =========================
@st.cache_resource
def load_eucs_model():
    model_path = BASE_DIR / "eucs_model.pkl"
    return joblib.load(model_path)

@st.cache_resource
def load_sentiment_pipe():
    model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)

    pipe = pipeline(
        task="sentiment-analysis",
        model=mdl,
        tokenizer=tok,
        truncation=True,
        max_length=512,
        device=-1  # CPU
    )
    return pipe, mdl

model_eucs = load_eucs_model()
sentiment_pipe, sentiment_mdl = load_sentiment_pipe()

# =========================
# HELPERS
# =========================
def normalize_sent_label(raw_label: str) -> str:
    lab = str(raw_label).lower()
    if lab.startswith("label_"):
        try:
            idx = int(lab.split("_")[-1])
            return str(sentiment_mdl.config.id2label[idx]).lower()
        except Exception:
            return lab
    return lab

def predict_eucs(texts):
    pred = model_eucs.predict(texts)  # (n, 5) 0/1
    pred_df = pd.DataFrame(pred, columns=DIMS)
    labels = pred_df.apply(lambda r: [d for d in DIMS if int(r[d]) == 1], axis=1).tolist()
    return pred_df, labels

def predict_sentiment(texts):
    preds = sentiment_pipe(texts, batch_size=16)
    labels = [normalize_sent_label(p["label"]) for p in preds]
    out = pd.DataFrame({
        "label_sentimen": labels,
        "confidence_score": [float(p["score"]) for p in preds],
    })
    out["sentiment_score"] = out["label_sentimen"].map(SENTIMENT_MAP).fillna(2).astype(int)
    return out

RECO_RULES = {
    "Content": "Perjelas informasi & detail fitur. Tambahkan panduan/FAQ untuk fitur utama (tabungan, transfer, deposito, pinjaman).",
    "Accuracy": "Prioritaskan stabilitas sistem: perbaiki bug/error, OTP/verifikasi, keamanan transaksi, dan validasi data.",
    "Format": "Benahi UI/UX: rapikan layout, konsistensi ikon/warna, tingkatkan keterbacaan dan navigasi.",
    "Ease_of_Use": "Sederhanakan alur: login/daftar, minim langkah, perjelas instruksi, tingkatkan kemudahan penggunaan.",
    "Timeliness": "Optimasi performa: kurangi loading/lemot/delay, perbaiki server/maintenance, tingkatkan respons aplikasi."
}

def build_recommendations(sent_label, eucs_labels):
    if sent_label != "negative":
        return []
    recs = [RECO_RULES[d] for d in eucs_labels if d in RECO_RULES]
    if not recs:
        recs = ["Lakukan evaluasi menyeluruh terhadap keluhan pengguna dan cek log error untuk menemukan penyebab utama."]
    return recs

# =========================
# UI
# =========================
st.title("💬 Comment Sentiment & EUCS Analysis Dashboard")

mode = st.sidebar.selectbox("Choose Analysis Type", ["Single Comment Analysis", "Batch Analysis (Excel/CSV)"])

if mode == "Single Comment Analysis":
    st.subheader("🔎 Single Comment Analysis")
    text = st.text_area("Enter your comment for analysis:")

    if st.button("Analyze Comment"):
        if not text.strip():
            st.warning("Komentar masih kosong.")
        else:
            sent_df = predict_sentiment([text])
            sent_label = sent_df.loc[0, "label_sentimen"]
            conf = float(sent_df.loc[0, "confidence_score"])

            pred_df, labels = predict_eucs([text])
            eucs_labels = labels[0]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 😊 Sentiment Analysis")
                st.write(f"**Sentiment:** {sent_label.upper()}")
                st.write(f"**Confidence:** {conf:.4f}")
                st.progress(min(conf, 1.0))

            with col2:
                st.markdown("### 🧾 EUCS Dimensions")
                if eucs_labels:
                    st.write(", ".join(eucs_labels))
                else:
                    st.write("- (Tidak terdeteksi dimensi EUCS)")

            st.markdown("### 💡 Improvement Recommendations")
            recs = build_recommendations(sent_label, eucs_labels)
            if recs:
                for i, r in enumerate(recs, 1):
                    st.write(f"{i}. {r}")
            else:
                st.write("Tidak ada rekomendasi karena sentimen tidak negatif.")

else:
    st.subheader("📊 Batch Analysis from Excel/CSV File")
    file = st.file_uploader("Upload Excel/CSV file with comments", type=["csv", "xlsx"])

    if file is not None:
        if file.name.endswith(".csv"):
            data = pd.read_csv(file)
        else:
            data = pd.read_excel(file)

        st.write("Preview data:")
        st.dataframe(data.head())

        text_col = st.selectbox("Pilih kolom komentar/ulasan:", options=data.columns)

        if st.button("Run Batch Analysis"):
            texts = data[text_col].fillna("").astype(str).tolist()

            sent_df = predict_sentiment(texts)
            pred_df, labels = predict_eucs(texts)

            result = pd.DataFrame({
                "ulasan": texts,
                "label_sentimen": sent_df["label_sentimen"],
                "confidence_score": sent_df["confidence_score"],
                "EUCS_Dimensions": [", ".join(l) if l else "-" for l in labels]
            })

            for d in DIMS:
                result[d] = pred_df[d].astype(int)

            st.markdown("### Summary")
            total = len(result)
            dist = result["label_sentimen"].value_counts(normalize=True) * 100

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Comments", total)
            c2.metric("Positive (%)", f"{dist.get('positive', 0):.1f}%")
            c3.metric("Neutral (%)", f"{dist.get('neutral', 0):.1f}%")
            c4.metric("Negative (%)", f"{dist.get('negative', 0):.1f}%")

            st.markdown("### Detailed Results")
            st.dataframe(result)

            st.download_button(
                "Download Results as CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="batch_results.csv",
                mime="text/csv"
            )
