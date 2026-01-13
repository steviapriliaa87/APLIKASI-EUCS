import streamlit as st
import pandas as pd
import numpy as np
import joblib

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Comment Sentiment & EUCS Dashboard",
    layout="wide"
)

DIMS = ["Content", "Accuracy", "Format", "Ease_of_Use", "Timeliness"]
SENTIMENT_MAP = {"negative": 1, "neutral": 2, "positive": 3}

# =========================
# LOAD MODELS (CACHE)
# =========================
@st.cache_resource
def load_eucs_model():
    return joblib.load("eucs_model.pkl")

@st.cache_resource
def load_sentiment_pipe():
    model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline(
        task="sentiment-analysis",
        model=mdl,
        tokenizer=tok,
        truncation=True,
        max_length=512,
        device=-1  # CPU biar stabil di streamlit cloud
    )

model_eucs = load_eucs_model()
sentiment_pipe = load_sentiment_pipe()

# =========================
# HELPERS
# =========================
def predict_eucs(texts):
    pred = model_eucs.predict(texts)  # output array 0/1 shape (n,5)
    pred_df = pd.DataFrame(pred, columns=DIMS)

    labels = pred_df.apply(
        lambda row: [d for d in DIMS if row[d] == 1], axis=1
    ).tolist()
    return pred_df, labels

def predict_sentiment(texts):
    # handle list input
    preds = sentiment_pipe(texts, batch_size=16)
    out = pd.DataFrame({
        "label_sentimen": [p["label"].lower() for p in preds],
        "confidence_score": [p["score"] for p in preds]
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
    # rekomendasi hanya kalau NEGATIVE
    if sent_label != "negative":
        return []
    recs = []
    for dim in eucs_labels:
        if dim in RECO_RULES:
            recs.append(RECO_RULES[dim])
    # kalau negatif tapi model gak deteksi dimensi, kasih general
    if not recs:
        recs = ["Lakukan evaluasi menyeluruh terhadap keluhan pengguna dan cek log error untuk menemukan penyebab utama."]
    return recs

# =========================
# UI
# =========================
st.title("💬 Comment Sentiment & EUCS Analysis Dashboard")

mode = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["Single Comment Analysis", "Batch Analysis (Excel/CSV)"]
)

# =========================
# SINGLE COMMENT
# =========================
if mode == "Single Comment Analysis":
    st.subheader("🔎 Single Comment Analysis")
    text = st.text_area("Enter your comment for analysis:")

    if st.button("Analyze Comment"):
        if not text.strip():
            st.warning("Komentar masih kosong.")
        else:
            # 1) sentiment
            sent_df = predict_sentiment([text])
            sent_label = sent_df.loc[0, "label_sentimen"]
            conf = float(sent_df.loc[0, "confidence_score"])

            # 2) eucs
            pred_df, labels = predict_eucs([text])
            eucs_labels = labels[0]

            # SHOW RESULTS
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 😊 Sentiment Analysis")
                st.write(f"**Sentiment:** {sent_label.upper()}")
                st.write(f"**Confidence:** {conf:.4f}")
                st.progress(min(conf, 1.0))

            with col2:
                st.markdown("### 🧾 EUCS Dimensions")
                if eucs_labels:
                    for d in eucs_labels:
                        st.write(f"- {d}")
                else:
                    st.write("- (Tidak terdeteksi dimensi EUCS)")

            # RECOMMENDATIONS
            st.markdown("### 💡 Improvement Recommendations")
            recs = build_recommendations(sent_label, eucs_labels)
            if recs:
                for i, r in enumerate(recs, 1):
                    st.write(f"{i}. {r}")
            else:
                st.write("Tidak ada rekomendasi karena sentimen tidak negatif.")

# =========================
# BATCH
# =========================
else:
    st.subheader("📊 Batch Analysis from Excel/CSV File")
    file = st.file_uploader("Upload Excel/CSV file with comments", type=["csv", "xlsx"])

    if file is not None:
        # READ FILE
        if file.name.endswith(".csv"):
            data = pd.read_csv(file)
        else:
            data = pd.read_excel(file)

        st.write("Preview data:")
        st.dataframe(data.head())

        # pilih kolom teks
        text_col = st.selectbox("Pilih kolom komentar/ulasan:", options=data.columns)

        if st.button("Run Batch Analysis"):
            texts = data[text_col].fillna("").astype(str).tolist()

            # sentiment + eucs
            sent_df = predict_sentiment(texts)
            pred_df, labels = predict_eucs(texts)

            # build final result
            result = pd.DataFrame({
                "ulasan": texts,
                "label_sentimen": sent_df["label_sentimen"],
                "confidence_score": sent_df["confidence_score"],
                "EUCS_Dimensions": [", ".join(l) if l else "-" for l in labels]
            })

            # add binary columns
            for d in DIMS:
                result[d] = pred_df[d].astype(int)

            # simple summary
            st.markdown("### Summary")
            total = len(result)
            dist = result["label_sentimen"].value_counts(normalize=True) * 100
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Comments", total)
            c2.metric("Positive (%)", f"{dist.get('positive',0):.1f}%")
            c3.metric("Negative (%)", f"{dist.get('negative',0):.1f}%")

            st.markdown("### Detailed Results")
            st.dataframe(result)

            # download
            st.download_button(
                "Download Results as CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="batch_results.csv",
                mime="text/csv"
            )
