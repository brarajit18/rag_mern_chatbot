import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

# Setting Up the Environment (Replace with your API key)
os.environ["OPENAI_API_KEY"] = "sk-rag-qa-serve-5qvva7Z07YgdqNoemNguT3BlbkFJLV5ytJavaFUVjRzRuGV4"  # Replace with your API key

# 1. Load the PDF Document
embeddings = OpenAIEmbeddings()  # Create an object for generating embeddings
llm = ChatOpenAI()              # Create an object to interact with the OpenAI API

pdf_loader = PyPDFLoader('Pro-MERN-Stack-Development-Express.pdf')
docs = pdf_loader.load()         # Load the questions from the PDF

# 2. Prepare the Chat Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# This template defines the format for prompting the LLM with context and a question.

# 3. Split the Text into Individual Questions
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# This step splits the loaded documents (likely containing multiple questions) into separate documents, each containing a single question.

# 4. Create Document Embeddings
vector = FAISS.from_documents(documents, embeddings)

# This line generates dense vector representations (embeddings) for each question. These embeddings capture the semantic meaning of the text and help retrieve relevant questions.

# 5. Build the Retrieval Chain
retriever = vector.as_retriever()
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Here, we create two chains:
#   - Retrieval Chain: This retrieves the most relevant question and its context based on the user's input question using the document embeddings.
#   - Document Chain: This chain uses the LLM to answer the user's question based on the retrieved context.

# 6. User Input and Response Generation
response = retrieval_chain.invoke({"input": "What are pure functions in javascript?"})
print(response["answer"])

# This prompts the user for a question, retrieves the most relevant question and context from the document, and then uses the LLM to answer the user's question based on that context. Finally, it prints the LLM's generated answer.