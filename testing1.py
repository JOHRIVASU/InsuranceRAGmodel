import streamlit as st
import pandas as pd
import os
import torch
import numpy as np
from datetime import datetime

try:
    from transformers import pipeline
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "transformers"])
    from transformers import pipeline

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

PDF_PATH = "C:/Users/VASU/Downloads/Sample HI Policy.pdf"
CSV_PATH = "C:/Users/VASU/Downloads/RAG_Test_Questions.csv"

st.set_page_config(page_title="PolicyGaido - Insurance Q&A", page_icon="📝", layout="wide")
st.title("Insurance Policy Q&A Assistant")
st.markdown("Ask questions about your insurance policy or select from predefined questions.")

if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vector_store = None
    st.session_state.questions = []
    st.session_state.evaluation_history = []

with st.sidebar:
    st.header("Configuration")
    model_option = st.selectbox("Select Language Model", ["BERT-for-QA", "DistilBERT-for-QA"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Running on: {device}")
    
    if st.button("Initialize System"):
        with st.spinner("Initializing Q&A system..."):
            st.session_state.initialized = True
    
    with st.expander("Performance Metrics"):
        if st.session_state.evaluation_history:
            total_queries = len(st.session_state.evaluation_history)
            avg_confidence = np.mean([eval_data['confidence'] for eval_data in st.session_state.evaluation_history])
            avg_relevance = np.mean([eval_data['relevance'] for eval_data in st.session_state.evaluation_history])
            
            st.metric("Total Queries", total_queries)
            st.metric("Avg. Confidence", f"{avg_confidence:.2f}%")
            st.metric("Avg. Relevance", f"{avg_relevance:.2f}%")
            
            if st.button("Clear History"):
                st.session_state.evaluation_history = []
                st.experimental_rerun()

@st.cache_data
def load_questions(csv_path):
    encodings = ["utf-8", "latin-1", "ISO-8859-1", "cp1252"]
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            # Check for both "Question" and "Questions" columns
            if "Question" in df.columns:
                return df["Question"].dropna().tolist()
            elif "Questions" in df.columns:
                return df["Questions"].dropna().tolist()
        except UnicodeDecodeError:
            continue 
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return []
    return []

if os.path.exists(CSV_PATH):
    st.session_state.questions = load_questions(CSV_PATH)
    if not st.session_state.questions:
        st.warning(f"No questions found in {CSV_PATH}. Make sure the file has a 'Questions' column.")

@st.cache_resource
def process_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': device})
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

if st.session_state.initialized and not st.session_state.vector_store:
    with st.spinner("Processing policy document..."):
        st.session_state.vector_store = process_pdf(PDF_PATH)

def get_answer(question, model_name):
    if st.session_state.vector_store is None:
        return "System not initialized. Please initialize first.", [], 0
    
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(question)
    context = " ".join([doc.page_content for doc in docs])
    
    if not context:
        return "No relevant information found in the policy document.", [], 0
    
    qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2" if model_name == "BERT-for-QA" else "distilbert-base-cased-distilled-squad", tokenizer="deepset/bert-base-cased-squad2", device=0 if torch.cuda.is_available() else -1)
    result = qa_pipeline(question=question, context=context)
    
    # Calculate semantic similarity between question and context as a relevance proxy
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': device})
    question_embedding = embeddings.embed_query(question)
    context_embedding = embeddings.embed_query(context[:1000])  # Use first 1000 chars to avoid token limits
    
    # Compute cosine similarity
    similarity = np.dot(question_embedding, context_embedding) / (np.linalg.norm(question_embedding) * np.linalg.norm(context_embedding))
    relevance_score = float(similarity * 100)
    
    return result["answer"], docs, result["score"], relevance_score

def evaluate_answer(answer, docs, confidence, relevance):
    # Count potentially hallucinatory indicators
    hallucination_indicators = 0
    
    # Check if answer contains content not found in supporting docs
    answer_found = False
    answer_words = set(answer.lower().split())
    
    if len(answer_words) > 0:
        for doc in docs:
            doc_content = doc.page_content.lower()
            overlap_count = sum(1 for word in answer_words if word in doc_content)
            if overlap_count / len(answer_words) > 0.3:  # At least 30% of answer words are in document
                answer_found = True
                break
    
    if not answer_found and len(answer_words) > 3:  # Only count if the answer is substantive
        hallucination_indicators += 1
    
    # Check for hedging language that might indicate uncertainty
    hedging_phrases = ["i think", "probably", "likely", "may", "might", "could be", "possibly", "perhaps"]
    if any(phrase in answer.lower() for phrase in hedging_phrases):
        hallucination_indicators += 1
    
    # Return hallucination risk score (0-100)
    hallucination_risk = min(100, hallucination_indicators * 50)
    
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confidence": confidence * 100,  # Convert to percentage
        "relevance": relevance,
        "hallucination_risk": hallucination_risk
    }

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Ask a Question")
    question_method = st.radio("Choose question input method:", ["Predefined", "Custom"])
    
    question = ""
    if question_method == "Predefined" and st.session_state.questions:
        question = st.selectbox("Select a question:", st.session_state.questions)
        if not st.session_state.questions:
            st.info("No predefined questions available. Please check your CSV file.")
    elif question_method == "Custom":
        question = st.text_area("Enter your question:")
    
    if question and st.button("Ask Question"):
        with st.spinner("Generating answer..."):
            answer, docs, confidence, relevance = get_answer(question, model_option)
            evaluation = evaluate_answer(answer, docs, confidence, relevance)
            st.session_state.evaluation_history.append(evaluation)
            st.session_state["last_answer"] = (question, answer, docs, evaluation)

with col2:
    st.subheader("Answer")
    if "last_answer" in st.session_state:
        question, answer, docs, evaluation = st.session_state["last_answer"]
        st.markdown(f"**Question:** {question}")
        st.markdown(f"**Answer:** {answer}")
        
        # Display evaluation metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Confidence", f"{evaluation['confidence']:.1f}%", 
                      delta=f"{evaluation['confidence']-50:.1f}" if evaluation['confidence'] > 50 else f"{evaluation['confidence']-50:.1f}")
        with col_b:
            st.metric("Relevance", f"{evaluation['relevance']:.1f}%",
                      delta=f"{evaluation['relevance']-50:.1f}" if evaluation['relevance'] > 50 else f"{evaluation['relevance']-50:.1f}")
        with col_c:
            st.metric("Hallucination Risk", f"{evaluation['hallucination_risk']:.1f}%",
                      delta=f"{-evaluation['hallucination_risk']+50:.1f}" if evaluation['hallucination_risk'] < 50 else f"{-evaluation['hallucination_risk']+50:.1f}", 
                      delta_color="inverse")
        
        with st.expander("View Source Information"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Source {i+1}:** {doc.page_content[:500]}...")

# History and statistics section
st.divider()
st.subheader("Evaluation History")

if st.session_state.evaluation_history:
    history_df = pd.DataFrame(st.session_state.evaluation_history)
    
    # Display summary statistics
    st.subheader("Performance Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg. Confidence", f"{history_df['confidence'].mean():.1f}%")
    with col2:
        st.metric("Avg. Relevance", f"{history_df['relevance'].mean():.1f}%")
    with col3:
        st.metric("Avg. Hallucination Risk", f"{history_df['hallucination_risk'].mean():.1f}%")
    
    # Show history table
    st.dataframe(history_df)
else:
    st.info("No evaluation history available yet. Ask some questions to build history.")

st.divider()
st.caption("PolicyGaido Insurance Q&A Assistant | Built with Streamlit, Transformers, and FAISS (By Vasu Johri)")