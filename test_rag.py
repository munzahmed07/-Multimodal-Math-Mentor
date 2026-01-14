
from src.rag import get_retriever

retriever = get_retriever()
docs = retriever.invoke("What is the derivative of x^n?")

print("\n--- RETRIEVED DOCS ---")
for doc in docs:
    print(f"[SOURCE]: {doc.metadata['source']}")
    print(f"[CONTENT]: {doc.page_content}\n")
