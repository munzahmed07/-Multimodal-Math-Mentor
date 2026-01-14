import os
import shutil
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv(find_dotenv())

DB_PATH = "./chroma_db"
KB_PATH = "./knowledge_base"


def initialize_vector_store():
    """Ingests documents and creates the Vector DB."""
    print(" --- RAG INIT STARTED ---")

    if not os.path.exists(KB_PATH):
        os.makedirs(KB_PATH)
        print(f" Created missing folder: {KB_PATH}")
        return

   
    loader = DirectoryLoader(KB_PATH, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()

    if not docs:
        print(" No documents found in knowledge_base/")
        return
    print(f"   Found {len(docs)} documents.")

    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("   Cleared old database.")

    print("   Creating embeddings...")

    
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )

    
    vectorstore.add_documents(documents=splits)

    print(f" SUCCESS: Vector store created at {DB_PATH}")


def get_retriever():
    """Returns the retriever for the agents."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("Run 'python src/rag.py' first!")

    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


if __name__ == "__main__":
    initialize_vector_store()
