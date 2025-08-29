
# RAG helper using LangChain, Chroma, and OpenAI
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

class RAGChromaHelper:
    def __init__(self, data_dir='data', persist_directory='chroma_db'):
        self.data_dir = data_dir
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY env var
        self._ensure_index()

    def _load_documents(self):
        docs = []
        if not os.path.exists(self.data_dir):
            return docs
        for fname in sorted(os.listdir(self.data_dir)):
            if fname.endswith('.txt'):
                loader = TextLoader(os.path.join(self.data_dir, fname), encoding='utf-8')
                docs.extend(loader.load())
        return docs

    def _ensure_index(self):
        # If a persisted Chroma collection exists, load it; otherwise build it from data/
        try:
            # attempt to load existing collection
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            # If collection empty, ingest
            # Access private API carefully; fall back to building if empty
            if hasattr(self.vectorstore, '_collection') and self.vectorstore._collection.count() == 0:
                raise Exception('empty')
        except Exception:
            docs = self._load_documents()
            if not docs:
                # empty placeholder
                texts = ["No documents available."]
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = []
                for d in docs:
                    parts = splitter.split_text(d.page_content)
                    texts.extend(parts)
            self.vectorstore = Chroma.from_texts(texts, embedding=self.embeddings, persist_directory=self.persist_directory)
            # persist is automatic for Chroma with persist_directory
        # Note: for production consider a hosted vector DB

    def answer_question(self, question, k=4):
        llm = OpenAI(temperature=0)
        retriever = self.vectorstore.as_retriever(search_kwargs={'k': k})
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type='map_reduce', retriever=retriever)
        res = qa.run(question)
        return res
