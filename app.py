import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# 📄 Dokumentumok
docs = [
    "The RAG model combines retrieval and generation.",
    "Google Colab is a cloud-based Python environment.",
    "FAISS is a fast vector search library.",
    "LangChain helps build LLM-based applications.",
    "Python is a programming language",
    "We use for this project lot of python library such as numpy, pandas, matplotlib, seaborn"
]

# ⚙️ Cache-elt embedding modell
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# 📦 Cache-elt vector store
@st.cache_resource
def build_vector_store(docs, embeddings):
    return FAISS.from_texts(docs, embeddings)

db = build_vector_store(docs, embeddings)

# 🤖 Cache-elt LLM pipeline
@st.cache_resource
def load_llm():
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=100)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# 🧠 Prompt sablon
prompt_template = """Answer the question based on the context.

{context}

Question: {question}
Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# 🔍 Cache-elt QA lánc
@st.cache_resource
def build_qa_chain(llm, db, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

qa = build_qa_chain(llm, db, prompt)

# 🎯 Streamlit UI
st.title("🔍 LangChain QA App")
st.write("Kérdezz bármit a projekt dokumentumaiból!")

user_query = st.text_input("Írd be a kérdésed:", placeholder="Pl. which library use for this project")

if user_query:
    with st.spinner("Gondolkodom..."):
        answer = qa.run(user_query)

    st.success("Válasz:")
    st.write(answer)
