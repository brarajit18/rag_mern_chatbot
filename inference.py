import os
import pickle
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# Setting Up the Environment (Replace with your API key)
os.environ["OPENAI_API_KEY"] = "sk-rag-qa-serve-5qvva7Z07YgdqNoemNguT3BlbkFJLV5ytJavaFUVjRzRuGV4"  # Replace with your API key

# 0. Configuration
embeddings = OpenAIEmbeddings()  # Create an object for generating embeddings
llm = ChatOpenAI()              # Create an object to interact with the OpenAI API

# 1. Load precomputed embeddings from a file (replace with your actual file path)
precomputed_embeddings = FAISS.load_local("./my_vectors/my_vector_store", embeddings, allow_dangerous_deserialization=True)

# 2. Prepare the Chat Prompt Template
# This template defines the format for prompting the LLM with context and a question.
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")
    
# 3. Build the Retrieval Chain
retriever = precomputed_embeddings.as_retriever()
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
# Here, we've created two chains:
#   - Retrieval Chain: This retrieves the most relevant question and its context based on the user's input question using the document embeddings.
#   - Document Chain: This chain uses the LLM to answer the user's question based on the retrieved context.

# 4. User Input and Response Generation
response = retrieval_chain.invoke({"input": "What are pure functions in javascript?"})
print(response["answer"])

# This prompts the user for a question, retrieves the most relevant question and context from the document, and then uses the LLM to answer the user's question based on that context. Finally, it prints the LLM's generated answer.