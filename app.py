import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# Dokumentumok
docs = [
    "The RAG model combines retrieval and generation.",
    "Google Colab is a cloud-based Python environment.",
    "FAISS is a fast vector search library.",
    "LangChain helps build LLM-based applications.",
    "Python is a programming language",
    "We use for this project lot of python library such as numpy, pandas, matplotlib, seaborn"
]

# Embedding modell
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store
@st.cache_resource
def load_vectorstore(embeddings):
    return FAISS.from_texts(docs, embeddings)

# LLM pipeline
@st.cache_resource
def load_llm():
    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=100
    )
    return HuggingFacePipeline(pipeline=llm_pipeline)

# Prompt sablon
prompt_template = """Answer the question based on the context.

{context}

Question: {question}
Answer:"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# QA lánc
@st.cache_resource
def load_qa(llm, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

# Streamlit UI
st.title("🔍 LangChain QA App")
st.write("Kérdezz bármit a projekt dokumentumaiból!")

user_query = st.text_input("Írd be a kérdésed:", placeholder="Pl. which library use for this project")

if user_query:
    embeddings = load_embeddings()
    db = load_vectorstore(embeddings)
    llm = load_llm()
    qa = load_qa(llm, db)

    with st.spinner("Gondolkodom..."):
        answer = qa.run(user_query)

    st.success("Válasz:")
    st.write(answer)
