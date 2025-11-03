import streamlit as st
import spacy
import os
os.system("python -m spacy download en_core_web_sm")
from bs4 import BeautifulSoup
import joblib



Model=joblib.load('spam_classfier.joblib')
tfidf=joblib.load('tfidf_vec.joblib')

nlp=spacy.load('en_core_web_sm')
# Preprocessing the data 

# removing htlm tags
import bs4
from bs4 import BeautifulSoup

def remove_html(text):
    soup=BeautifulSoup(text,'html.parser')
    return soup.get_text(separator='',strip=True)

# remove punctuations
import string
exclude=string.punctuation

def remove_punctions(text):
    for word in exclude:
        text=text.replace(word,"")
    return text 


# lemitization tokenization urls and numbers
import spacy       

nlp=spacy.load('en_core_web_sm')

def Tokenization(text):
    cols=nlp(text)
    Tokens=[]
    for token in cols:
        if token.like_url:
            Tokens.append('url')
        elif token.like_num:
            Tokens.append('num')
        elif token.is_alpha and not  token.is_stop:
            Tokens.append(token.lemma_)
    return " ".join(Tokens)
        

def preprocess(text):
    if  not isinstance(text,str):
        return ""
     
    text=text.lower()
    text=remove_html(text)
    text=remove_punctions(text)
    return Tokenization(text)


st.set_page_config(page_title="Email Spam Classfier",page_icon="📩",layout='centered')

st.title('Email/SMS Spam Classfier')
st.write("Detect whether an SMS message is spam or not using a trained LightGBM model.")

user_input=st.text_area("Enter your message",height=150)

if st.button('Predict'):
    if user_input.strip() =="":
        st.warning("Please Enter a message to classify")
    else:
        pre_processed_text=preprocess(user_input)

        vec_input=tfidf.transform([pre_processed_text])

        prediction=Model.predict(vec_input)[0]

        if prediction  == 1:
            st.error("🚨 This message is **SPAM**!")
        else:
            st.success("✅ This message is **NOT SPAM**.")

st.markdown("-----")
st.caption("Made with ❤️ by Dhruv | Powered by LightGBM + Streamlit")