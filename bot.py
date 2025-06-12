import streamlit as st
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

DB_FAISS_PATH = "vectorstore/faiss_index"
@st.cache_resource
def get_vector_store():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db= FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return  db

def set_customer_prompt_template(CUSTOM_PROMPT_TEMPLATE):
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    return prompt

def load_llm(HUGGINGFACE_REPO_ID , HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        max_new_tokens=512,  # Adjust as needed
        huggingfacehub_api_token=HF_TOKEN 
    )
    return llm


def format_sources(docs):
    formatted = ""
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page_label", "N/A")
        title = doc.metadata.get("title", "Unknown Title")
        snippet = doc.page_content[:300].replace("\n", " ")  # Trim long text and remove newlines
        formatted += f"🔹 **Source {i}** (Page {page}, *{title}*):\n> {snippet}...\n\n"
    return formatted

def main():
    st.title("Ask Chatbot! ")

    if 'messages' not in st.session_state: # Check if messages list exists in session state
        st.session_state.messages = [] # Initialize messages list

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt=st.chat_input("Pass your prompt  here")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})


        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of info  provided in the context to answer the user's question..
        If you do not know the answer, just say that you do not know, do not try to make up an answer.
        Dont provide anything out of the context.

        Context: {context}
        Question: {question}

        Start the answer directly with no small talk.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")
        


        try:
            vectorstore=get_vector_store()
            if vectorstore is None:
                st.error("Vector store not found. Please ensure it is created and available.")
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID , HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={
                            "prompt": set_customer_prompt_template(CUSTOM_PROMPT_TEMPLATE)
                }   
                )
            response=qa_chain.invoke({"query": prompt})

            result=response["result"]
            source_documents = response["source_documents"]
            result_to_show = result + "\n\n### 📚 Source Documents:\n" + format_sources(source_documents)

    

        #response="Hi! Im a medibot"
            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})
        except Exception as e:
            st.error(f"An error occurred: {e}")
            

if __name__ == "__main__":
    main()
    