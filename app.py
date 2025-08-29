import os
from flask import Flask, request, render_template, jsonify
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
# 1. Flask setup
import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Initialize embeddings + vectorstore
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectordb = Chroma(persist_directory="chroma_store", embedding_function=embeddings)

# Retrieval
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini", temperature=0.3)

# Prompt template for RAG
rag_prompt = PromptTemplate(
    template="""
You are a helpful Chartered Accountant assistant in Mumbai.
Always answer professionally and clearly. 

If the information is available in the context, use it. 
If the context is empty or not enough, still try to answer from your general CA knowledge
(taxation, GST, income tax slabs, compliance, company formation, etc).
If you are unsure, guide the client politely to consult a CA instead of saying "I don't know".

Context: {context}
Question: {question}

Answer:
""",
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": rag_prompt}
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.form.get("question")
        if not question:
            return jsonify({"error": "No question provided"}), 400

        response = qa_chain.run(question)

        if not response or response.strip() == "":
            response = "I couldn't find a direct answer, but I recommend consulting a Chartered Accountant for this matter."

        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
