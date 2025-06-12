#Setup llm with Huggingafcae
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

HF_TOKEN=os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(HUGGINGFACE_REPO_ID):
    """
    Load the LLM from HuggingFace.
    """
    
    llm=HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        max_new_tokens=512,  # ✅ Moved here — not inside model_kwargs
        huggingfacehub_api_token=HF_TOKEN)
    return llm

#step2 :Connect with llm and create chain

DB_FAISS_PATH = "vectorstore/faiss_index"
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of info  provided in the context to answer the user's question..
If you do not know the answer, just say that you do not know, do not try to make up an answer.
Dont provide anything out of the context.

Context: {context}
Question: {question}

Start the answer directly with no small talk.
"""

def set_customer_prompt_template(CUSTOM_PROMPT_TEMPLATE):
    
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    return prompt

#load databse
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db=FAISS.load_local(DB_FAISS_PATH, embedding_model , allow_dangerous_deserialization=True)

#CRetea  qa chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": set_customer_prompt_template(CUSTOM_PROMPT_TEMPLATE)
    }
)

#Now invoke witha a single query
user_query=input("Enter your query: ")
response = qa_chain.invoke({"query": user_query})
print("RESULT:"  ,response['result'])
print("SOURCE DOCUMENTS:" , response['source_documents'])