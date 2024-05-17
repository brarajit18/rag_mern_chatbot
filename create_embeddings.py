import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setting Up the Environment (Replace with your API key)
os.environ["OPENAI_API_KEY"] = "sk-rag-qa-serve-5qvva7Z07YgdqNoemNguT3BlbkFJLV5ytJavaFUVjRzRuGV4"  # Replace with your API key

# 0. Configuration
embeddings = OpenAIEmbeddings()  # Create an object for generating embeddings

# 1. Read from the pdf file
pdf_loader = PyPDFLoader('Pro-MERN-Stack-Development-Express.pdf')
docs = pdf_loader.load()         # Load the questions from the PDF

# 2. Split the Text into Individual Questions
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# 3. Create Document Embeddings
vectorindex = FAISS.from_documents(documents, embeddings)

# 4. 
# Save the vector store locally
vectorindex.save_local("./my_vectors/my_vector_store")

# with open("precomputed_embeddings.pkl", "wb") as f:
#     pickle.dump(vectorindex, f)
    