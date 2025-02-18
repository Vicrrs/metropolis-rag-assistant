import os
import json
from bs4 import BeautifulSoup

import faiss

# Importações do langchain “core”:
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Importações do langchain_community:
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp

# Importação de embeddings do langchain-huggingface:
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------
# 1. Funções de extração de texto para cada tipo de arquivo
# ---------------------------------------------------------

def extract_text_from_txt(filepath):
    """Lê arquivo .txt e retorna o conteúdo como string."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Erro ao ler TXT {filepath}: {e}")
        return ""

def extract_text_from_html(filepath):
    """Lê arquivo .html e retorna todo o texto (sem tags) como string."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, 'html.parser')
            return soup.get_text(strip=True)
    except Exception as e:
        print(f"Erro ao ler HTML {filepath}: {e}")
        return ""

def extract_text_general(filepath):
    """
    Lê arquivos texto como .py, .md, .yml e retorna seu conteúdo.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Erro ao ler arquivo texto {filepath}: {e}")
        return ""

def extract_text_from_ipynb(filepath):
    """
    Lê um notebook Jupyter (.ipynb), que é JSON, e extrai o texto de cada célula.
    """
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
    except Exception as e:
        print(f"Erro ao ler IPYNB {filepath}: {e}")
        return ""

# ---------------------------------------------------------
# 2. Carrega documentos a partir de uma pasta
# ---------------------------------------------------------

def load_documents_from_folder(folder_path):
    """
    Percorre recursivamente a pasta, extraindo texto de cada arquivo
    suportado e retorna uma lista de dicionários:
    [{ "filepath": <caminho>, "text": <conteúdo> }, ...]
    """
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

# ---------------------------------------------------------
# 3. Função para dividir texto em chunks
# ---------------------------------------------------------

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    folder_path = "/home/vicrrs/projetos/meus_projetos/metropolis-rag-assistant/docs_scraper/text_output"
    folder_path01 = "/home/vicrrs/projetos/meus_projetos/metropolis-rag-assistant/deepstream"

    documents_folder0 = load_documents_from_folder(folder_path)
    documents_folder1 = load_documents_from_folder(folder_path01)
    documents = documents_folder0 + documents_folder1
    print(f"Total de documentos carregados: {len(documents)}")

    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size=1000, chunk_overlap=100)
        for ch in chunks:
            all_chunks.append({
                "text": ch,
                "metadata": {"source": doc["filepath"]}
            })

    print(f"Total de chunks: {len(all_chunks)}")

    # Carrega embeddings na GPU (se suportado)
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda"}
    )

    texts = [ch["text"] for ch in all_chunks]
    metadatas = [ch["metadata"] for ch in all_chunks]

    db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    faiss_index_path = "meu_indice.faiss"
    faiss.write_index(db.index, faiss_index_path)
    print(f"Índice FAISS salvo em {faiss_index_path}")

    MODEL_PATH = "/home/vicrrs/.lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=8192,
        temperature=0.1,
        max_tokens=2048,
        n_gpu_layers=50,
        n_batch=512,
        verbose=False
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    template = """
Você é um assistente que só pode usar as passagens de texto fornecidas abaixo para responder.
Se a resposta não estiver neles, responda "Não sei".

Passagens relevantes:
{context}

Pergunta: {question}
Resposta:
"""
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    qa_chain_custom = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

    pergunta3 = "De tutorial de como elaborar pipeline deepstream, retorne apenas o codigo da pipeline para um modelo multcameras!"
    resposta3 = qa_chain_custom.invoke({"query": pergunta3})

    print("\nPergunta 3 (PromptTemplate):", pergunta3)
    print("Resposta 3:", resposta3["result"])


