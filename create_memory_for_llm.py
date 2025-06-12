#Step1:  Load raw pdf
#Step2: Create chunks
#Step3: Create embeddings
#Step4: Create vector store
#Step5: Store embedding in Faiss 
from dotenv import load_dotenv
import os

DATA_PATH = "data/" 
from collections import defaultdict
from langchain_community.document_loaders import PyPDFLoader  , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_pdf_files(data):
    """
    Load a PDF file and return its content as a list of documents.
    """
    loader = DirectoryLoader(data ,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents = loader.load()
    #print(f"Loaded {len(documents)} documents from {data}")
    return documents

documents = load_pdf_files(DATA_PATH)
grouped_docs = defaultdict(list)
for doc in documents:
    source = doc.metadata.get('source', 'Unknown')
    grouped_docs[source].append(doc)

# Step 3: Print the number of pages for each PDF
#print("\nPDF Page Breakdown:")
for pdf_file, pages in grouped_docs.items():
    print(f"  {pdf_file} - {len(pages)} pages")


# Step 4: Create chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Adjust chunk size as needed
        chunk_overlap=50,  # Adjust overlap as needed
        )
    text_chunks= text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
print(f"\nTotal number of chunks created: {len(text_chunks)}")


def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model
embedding_model = get_embedding_model()

# Step 5: Create vector store
DB_FAISS_PATH= "vectorstore/faiss_index"
db= FAISS.from_documents(text_chunks , embedding_model)
db.save_local(DB_FAISS_PATH)
print(" FAISS vector store created and saved at:", DB_FAISS_PATH)
