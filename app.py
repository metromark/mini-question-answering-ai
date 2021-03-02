import streamlit as st
from transformers import AutoTokenizer,AutoModelForQuestionAnswering
from transformers.pipelines import pipeline
import pandas as pd
from pathlib import Path

st.cache(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-small-finetuned-squadv2")
    model = AutoModelForQuestionAnswering.from_pretrained("mrm8488/bert-small-finetuned-squadv2")
    nlp_pipe = pipeline('question-answering',model=model,tokenizer=tokenizer)
    return nlp_pipe
npl_pipe = load_model()

st.header("QUAN.AI Lite")
st.text("Lighter version for Question-Answering AI for BARMM.")

question = st.text_input(label="Ask me anything. I'm small so I don't know the answer to everything but I can try.")
text = Path('data/barmm.txt').read_text()

if (not len(text)==0) and (not len(question)==0):
    x_dict = npl_pipe(context=text, question=question)
    st.text(x_dict['answer'])
    