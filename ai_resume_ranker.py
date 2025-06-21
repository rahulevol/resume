# AI-Powered Resume Ranker - Streamlit Version
# Objective: Rank resumes against a job description using NLP with Streamlit UI

import os
import subprocess
import spacy
import PyPDF2
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# Ensure SpaCy model is available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Helper function to extract text from uploaded PDF

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join(page.extract_text() or "" for page in reader.pages)

# Preprocess text using SpaCy

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Rank resumes based on job description

def rank_resumes(jd_text, resumes):
    corpus = [jd_text] + [r['text'] for r in resumes]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    for i, score in enumerate(scores):
        resumes[i]['score'] = round(score * 100, 2)
    return sorted(resumes, key=lambda x: x['score'], reverse=True)

# Streamlit UI
st.title("📄 AI-Powered Resume Ranker")
st.write("Upload a job description and candidate resumes to rank them based on relevance.")

jd_file = st.file_uploader("Upload Job Description (TXT)", type=["txt"])
resume_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

if jd_file and resume_files:
    jd_text = jd_file.read().decode("utf-8")
    jd_processed = preprocess(jd_text)

    resumes = []
    for file in resume_files:
        pdf_text = extract_text_from_pdf(file)
        resumes.append({"name": file.name, "text": preprocess(pdf_text)})

    ranked = rank_resumes(jd_processed, resumes)

    df = pd.DataFrame([{"Name": r['name'], "Score": r['score']} for r in ranked])
    st.subheader("📊 Ranked Resumes")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download HR Report (CSV)", csv, "hr_ranked_report.csv", "text/csv")
