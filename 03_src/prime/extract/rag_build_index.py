from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

SNIPPETS_DIR = "snippets"
DB_DIR = "chroma_db"

def load_docs():
    docs = []
    for p in sorted(Path(SNIPPETS_DIR).glob("*.txt")):
        paper_id = p.stem
        # one big doc per paper (we’ll chunk next)
        d = TextLoader(str(p), encoding="utf-8").load()
        for doc in d:
            doc.metadata.update({"paper_id": paper_id})
            docs.append(doc)
    return docs

def main():
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=800)
    chunks = splitter.split_documents(docs)

    embed = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embed,
        persist_directory=DB_DIR,
        collection_name="papers"
    )
    vectordb.persist()
    print(f"✅ Built vector DB with {vectordb._collection.count()} chunks at {DB_DIR}")

if __name__ == "__main__":
    main()
