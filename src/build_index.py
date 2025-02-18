import os
import json
import pickle
import faiss
from bs4 import BeautifulSoup

# LangChain "core"
from langchain.text_splitter import CharacterTextSplitter

# LangChain Community
from langchain_community.vectorstores import FAISS

# Embeddings HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings


def extract_text_from_txt(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""

def extract_text_from_html(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text(strip=True)
    except:
        return ""

def extract_text_general(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""

def extract_text_from_ipynb(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        extracted_text = []
        for cell in data.get("cells", []):
            cell_type = cell.get("cell_type", "")
            source_lines = cell.get("source", [])
            cell_text = "".join(source_lines)
            extracted_text.append(f"({cell_type}):\n{cell_text}")
        return "\n\n".join(extracted_text)
    except:
        return ""

def load_documents_from_folder(folder_path):
    supported_extensions = [".txt", ".html", ".py", ".md", ".yml", ".json", ".ipynb"]
    docs = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()

            text = ""
            if ext == ".txt":
                text = extract_text_from_txt(filepath)
            elif ext == ".html":
                text = extract_text_from_html(filepath)
            elif ext in [".py", ".md", ".yml", ".json"]:
                text = extract_text_general(filepath)
            elif ext == ".ipynb":
                text = extract_text_from_ipynb(filepath)
            else:
                continue

            if text.strip():
                docs.append({"filepath": filepath, "text": text})
    return docs

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

def build_and_save_index(
    folder_paths,
    faiss_index_path,
    docstore_path,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"
):
    """
    Cria e salva o índice FAISS em 'faiss_index_path' e
    também salva docstore + index_to_docstore_id em 'docstore_path'.
    """
    # 1. Carrega documentos
    all_docs = []
    for fp in folder_paths:
        all_docs += load_documents_from_folder(fp)
    print(f"Total de documentos carregados: {len(all_docs)}")

    # 2. Divide em chunks
    all_chunks = []
    for doc in all_docs:
        chunks = chunk_text(doc["text"], chunk_size=1000, chunk_overlap=100)
        for ch in chunks:
            all_chunks.append({
                "text": ch,
                "metadata": {"source": doc["filepath"]}
            })
    print(f"Total de chunks: {len(all_chunks)}")

    # 3. Carrega embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device}
    )

    # 4. Cria índice FAISS via from_texts e salva índice + docstore
    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    # Salva o índice numericamente
    faiss.write_index(db.index, faiss_index_path)
    print(f"Índice FAISS salvo em: {faiss_index_path}")

    # Salva docstore e index_to_docstore_id em um arquivo (via pickle)
    with open(docstore_path, "wb") as f:
        pickle.dump((db.docstore, db.index_to_docstore_id), f)
    print(f"Docstore salvo em: {docstore_path}")

if __name__ == "__main__":
    # Defina as pastas que contêm documentos
    folder_paths = [
        "/home/vicrrs/projetos/meus_projetos/metropolis-rag-assistant/docs_scraper/text_output",
        "/home/vicrrs/projetos/meus_projetos/metropolis-rag-assistant/deepstream"
    ]
    faiss_index_path = "meu_indice.faiss"
    docstore_path = "meu_docstore.pkl"
    
    # Se preferir CPU, troque device="cpu"
    build_and_save_index(
        folder_paths,
        faiss_index_path,
        docstore_path,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda"
    )
