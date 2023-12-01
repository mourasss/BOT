# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 12:45:39 2023

@author: EL AOUD
"""

import requests
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
import os
import tempfile
from langchain.document_loaders import PyPDFLoader
import streamlit as st
import huggingface_hub
from langchain.text_splitter import CharacterTextSplitter


def chunking_Embedding(text):
    #chunking
    text_splitter = CharacterTextSplitter(
       separator="\n",
       chunk_size=100,
       chunk_overlap=100,
       length_function=len
     )
    chunks=text_splitter.split_documents(text)
    #Embedding
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings()
    from langchain.vectorstores import FAISS
    db = FAISS.from_documents(chunks, embeddings)
    return db


st.set_page_config(layout="wide")
with st.sidebar:
    st.title("🤖 Mr Hadiouch")
    Api_token=st.text_input('Coller votre HuggingFace Api Token')
    uploaded_file = st.file_uploader("Charger an article", type=("pdf"))
os.environ["HUGGINGFACEHUB_API_TOKEN"] = Api_token

if "messages" not in st.session_state:
    st.session_state.messages=[{"role": "ai", "content": "Bonjour, veuillez entrer votre Token et charger un fichier dans le volet à gauche !"}]
if "DB" not in st.session_state:    
    st.session_state.DB=0
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
if uploaded_file is not None :
    if question := st.chat_input() :
        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("user"):
              st.write(question)

if uploaded_file is not None  and st.session_state.DB==0  : 
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_file_path, "wb") as temp_file:
           temp_file.write(uploaded_file.read())
           loader = PyPDFLoader(temp_file_path)
        loader = PyPDFLoader(temp_file_path)
        text=loader.load()
        #Chunking_embedding
        with st.sidebar:
            with st.spinner('Entrain de traiter le fichier'):
                db=chunking_Embedding(text)
            st.success('Terminé !')
        st.session_state['DB']=db
    
if st.session_state.messages[-1]["role"] != "ai":
        db=st.session_state['DB']
        llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length":512})
        chain = load_qa_chain(llm, chain_type="stuff")    
        docs = db.similarity_search(question)
        with st.chat_message("assistant"):
            response=chain.run(input_documents=docs, question=question) 
            placeholder = st.empty()
            full_response = ''
            for item in response :
                full_response+=item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
        st.session_state.messages.append({"role":"ai","content":full_response})
    # prompt = PromptTemplate(template=template, input_variables=["question"])
