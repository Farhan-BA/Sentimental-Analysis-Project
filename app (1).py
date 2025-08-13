
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title='SentiView', page_icon='🤖', layout='wide')

@st.cache_resource
def ensure_nltk():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    return True

ensure_nltk()

@st.cache_resource
def load_artifacts():
    model = joblib.load('sentiment_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    return model, vectorizer

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    return " ".join(lemmatized)

st.title('🤖 SentiView — IMDB Sentiment Analyzer')
tabs = st.tabs(['Predict','Batch Analyze','Model Insights','Dataset Explorer'])

with tabs[0]:
    st.subheader('Single Text Prediction')
    user_text = st.text_area('Enter text', 'The movie was absolutely fantastic, a true masterpiece!', height=160)
    if st.button('Analyze'):
        try:
            model, vectorizer = load_artifacts()
            cleaned = preprocess_text(user_text)
            X = vectorizer.transform([cleaned])
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            sentiment = 'Positive' if pred == 1 else 'Negative'
            conf = float(np.max(proba))
            col1, col2 = st.columns(2)
            with col1:
                st.metric('Sentiment', sentiment)
            with col2:
                st.metric('Confidence', f'{conf*100:.2f}%')
            st.progress(conf)
        except Exception as e:
            st.error(f'Error: {e}')

with tabs[1]:
    st.subheader('Batch Analyze CSV')
    st.caption('CSV must contain a column named "review"')
    up = st.file_uploader('Upload CSV', type=['csv'])
    if up is not None:
        df = pd.read_csv(up)
        if 'review' not in df.columns:
            st.error('Column "review" not found')
        else:
            model, vectorizer = load_artifacts()
            df['cleaned_review'] = df['review'].astype(str).apply(preprocess_text)
            X = vectorizer.transform(df['cleaned_review'])
            preds = model.predict(X)
            probas = model.predict_proba(X)[:,1]
            df['sentiment'] = np.where(preds==1,'Positive','Negative')
            df['confidence'] = probas
            st.dataframe(df[['review','sentiment','confidence']].head(50))
            st.bar_chart(df['sentiment'].value_counts())
            st.download_button('Download Results', df.to_csv(index=False).encode('utf-8'), file_name='sentiview_results.csv', mime='text/csv')

with tabs[2]:
    st.subheader('Model Insights')
    try:
        with open('metrics.json','r') as f:
            m = json.load(f)
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Accuracy', f"{m['accuracy']:.4f}")
        with col2:
            st.metric('ROC AUC', f"{m['roc_auc']:.4f}")
        st.write('Classification Report')
        cr = pd.DataFrame(m['classification_report']).transpose()
        st.dataframe(cr)
        st.write('Confusion Matrix')
        cm = np.array(m['confusion_matrix'])
        cm_df = pd.DataFrame(cm, index=['True Neg','True Pos'], columns=['Pred Neg','Pred Pos'])
        st.dataframe(cm_df)
        st.caption('Top Features')
        model, vectorizer = load_artifacts()
        coefs = model.coef_[0]
        feats = np.array(vectorizer.get_feature_names_out())
        top_pos_idx = np.argsort(coefs)[-20:][::-1]
        top_neg_idx = np.argsort(coefs)[:20]
        colp, coln = st.columns(2)
        with colp:
            st.write('Most Positive Tokens')
            st.table(pd.DataFrame({'token': feats[top_pos_idx], 'weight': coefs[top_pos_idx]}))
        with coln:
            st.write('Most Negative Tokens')
            st.table(pd.DataFrame({'token': feats[top_neg_idx], 'weight': coefs[top_neg_idx]}))
    except Exception as e:
        st.warning('Run training first to see metrics.')

with tabs[3]:
    st.subheader('Dataset Explorer')
    st.caption('Load the original IMDB dataset CSV to preview and filter.')
    up2 = st.file_uploader('Upload IMDB Dataset CSV', type=['csv'], key='explorer')
    if up2 is not None:
        df2 = pd.read_csv(up2)
        st.dataframe(df2.head(100))
        if 'review' in df2.columns and 'sentiment' in df2.columns:
            st.bar_chart(df2['sentiment'].value_counts())
