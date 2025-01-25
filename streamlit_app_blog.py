import streamlit as st
from pdf_qa import PdfQA
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil
from constants import *



# Streamlit app code
st.set_page_config(
    page_title='Q&A Bot for PDF',
    page_icon='🔖',
    layout='wide',
    initial_sidebar_state='auto',
)


if "pdf_qa_model" not in st.session_state:
    st.session_state["pdf_qa_model"]:PdfQA = PdfQA() ## Intialisation

## To cache resource across multiple session 
@st.cache_resource
def load_llm(llm,load_in_8bit):

    if llm == LLM_OPENAI_GPT35:
        pass
    elif llm == LLM_FLAN_T5_SMALL:
        return PdfQA.create_flan_t5_small(load_in_8bit)
    elif llm == LLM_FLAN_T5_BASE:
        return PdfQA.create_flan_t5_base(load_in_8bit)
    elif llm == LLM_FLAN_T5_LARGE:
        return PdfQA.create_flan_t5_large(load_in_8bit)
    elif llm == LLM_FASTCHAT_T5_XL:
        return PdfQA.create_fastchat_t5_xl(load_in_8bit)
    elif llm == LLM_FALCON_SMALL:
        return PdfQA.create_falcon_instruct_small(load_in_8bit)
    else:
        raise ValueError("Invalid LLM setting")

## To cache resource across multiple session
@st.cache_resource
def load_emb(emb):
    if emb == EMB_SBERT_MINILM:
        pass ##ChromaDB takes care
    else:
        raise ValueError("Invalid embedding setting")



st.title("PDF Q&A (Self hosted LLMs)")

with st.sidebar:
    emb = EMB_SBERT_MINILM
    llm = st.radio("**Select LLM Model**", [LLM_FASTCHAT_T5_XL, LLM_FLAN_T5_SMALL,LLM_FLAN_T5_BASE,LLM_FLAN_T5_LARGE,LLM_FLAN_T5_XL,LLM_FALCON_SMALL],index=2)
    load_in_8bit =  False
    pdf_file = st.file_uploader("**Upload PDF**", type="pdf")

    
    if st.button("Submit") and pdf_file is not None:
        with st.spinner(text="Uploading PDF and Generating Embeddings.."):
            with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                shutil.copyfileobj(pdf_file, tmp)
                tmp_path = Path(tmp.name)
                st.session_state["pdf_qa_model"].config = {
                    "pdf_path": str(tmp_path),
                    "embedding": emb,
                    "llm": llm,
                    "load_in_8bit": load_in_8bit
                }
                st.session_state["pdf_qa_model"].embedding = load_emb(emb)
                st.session_state["pdf_qa_model"].llm = load_llm(llm,load_in_8bit)        
                st.session_state["pdf_qa_model"].init_embeddings()
                st.session_state["pdf_qa_model"].init_models()
                st.session_state["pdf_qa_model"].vector_db_pdf()
                st.sidebar.success("PDF uploaded successfully")

question = st.text_input('Ask a question', 'What is this document?')

if st.button("Answer"):
    try:
        st.session_state["pdf_qa_model"].retreival_qa_chain()
        answer = st.session_state["pdf_qa_model"].answer_query(question)
        st.write(f"{answer}")
    except Exception as e:
        st.error(f"Error answering the question: {str(e)}")