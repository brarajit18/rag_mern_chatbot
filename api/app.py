from flask import Flask, request, jsonify
import os
import pickle
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# Initialize the Flask application
app = Flask(__name__)

# Setting Up the Environment (Replace with your API key)
os.environ["OPENAI_API_KEY"] = "sk-rag-qa-serve-5qvva7Z07YgdqNoemNguT3BlbkFJLV5ytJavaFUVjRzRuGV4"  # Replace with your API key

# 0. Configuration
embeddings = OpenAIEmbeddings()  # Create an object for generating embeddings
llm = ChatOpenAI()              # Create an object to interact with the OpenAI API

# 1. Load precomputed embeddings from a file (replace with your actual file path)
precomputed_embeddings = FAISS.load_local("./my_vectors/my_vector_store", embeddings, allow_dangerous_deserialization=True)

# 2. Prepare the Chat Prompt Template
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

# Define the API endpoint
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        # Get the user input from the request
        user_input = request.json.get('question')
        if not user_input:
            return jsonify({"error": "Question is required"}), 400
        
        # Generate the response using the retrieval chain
        response = retrieval_chain.invoke({"input": user_input})
        
        # Return the response as JSON
        return jsonify({"answer": response["answer"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# # Run the Flask application
# if __name__ == '__main__':
#     app.run(debug=True)
