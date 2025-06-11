from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os

# Load the document
def load_vectorstore():
    loader = TextLoader("app/data_explanation_section.txt", encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedder)

    return vectorstore

# Build the LLM and QA chain
def build_qa_chain():
    vectorstore = load_vectorstore()

    # Load hosted LLM from HuggingFaceHub
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-alpha",
        model_kwargs={"temperature": 0.6, "max_new_tokens": 256}
    )

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an AI assistant. Answer the question **only** based on the context below. Don't guess.

Context:
{context}

Question:
{question}

Answer:"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": custom_prompt}
    )

    return qa_chain

# Global chain instance
qa_chain = build_qa_chain()

def get_bot_answer(query: str) -> str:
    result = qa_chain({"query": query})
    return result["result"]
