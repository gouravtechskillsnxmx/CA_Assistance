import os
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# --- Embeddings + Vectorstore setup ---
# ⚠️ Ensure you have already ingested documents into "chroma_store/"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectordb = Chroma(persist_directory="chroma_store", embedding_function=embeddings)

# Retrieval
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# --- LLM setup ---
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",   # ✅ updated arg is 'model' not 'model_name'
    temperature=0.3
)

# --- Prompt template ---
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

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = None

        # 1. Try form
        if request.form:
            question = request.form.get("question")

        # 2. Try JSON
        if not question and request.is_json:
            data = request.get_json(silent=True)
            if data:
                question = data.get("question")

        # 3. Try query params (for testing in browser: /ask?question=hello)
        if not question:
            question = request.args.get("question")

        if not question:
            return jsonify({"error": "No question provided"}), 400

        print("Received question:", question)  # Debugging

        response = qa_chain.run(question)

        if not response or response.strip() == "":
            response = "I couldn't find a direct answer, but I recommend consulting a Chartered Accountant for this matter."

        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
