import os
import faiss
import pickle
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# "FAISS" e "LlamaCpp" da langchain_community
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp

# Embeddings HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings

def load_qa_chain(
    faiss_index_path="meu_indice.faiss",
    docstore_path="meu_docstore.pkl",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    model_path="/home/vicrrs/.lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
    device="cuda"
):
    """
    Carrega o índice FAISS + docstore e monta a chain de QA
    usando LlamaCpp como LLM.
    """
    print(f"Carregando índice FAISS de: {faiss_index_path}")
    index = faiss.read_index(faiss_index_path)

    # Carrega embeddings (mesmo modelo usado para criar o índice)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device}
    )

    # Carrega o docstore + index_to_docstore_id salvos em pickle
    with open(docstore_path, "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)

    # Reconstrói o VectorStore FAISS passando todos parâmetros
    db = FAISS(
        embeddings,             # embedding_function
        index,                  # faiss_index
        docstore,               # docstore
        index_to_docstore_id    # index_to_docstore_id
    )

    # Configura o LLM
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=8192,
        temperature=0.1,
        max_tokens=2048,
        n_gpu_layers=50,
        n_batch=512,
        verbose=False
    )

    # Cria o retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Prompt customizado (opcional)
    template = """
Você é um assistente que só pode usar as passagens de texto fornecidas abaixo para responder.
Se a resposta não estiver neles, responda "Não sei".

Passagens relevantes:
{context}

Pergunta: {question}
Resposta:
"""
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    # Monta a chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

def main():
    # Carrega a chain (não recria o índice, só reaproveita o salvo)
    qa_chain = load_qa_chain(
        faiss_index_path="meu_indice.faiss",
        docstore_path="meu_docstore.pkl",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        model_path="/home/vicrrs/.lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        device="cuda"  # ou "cpu"
    )

    print("=== Digite suas perguntas. 'sair' para encerrar. ===")
    while True:
        pergunta = input("\nPergunta: ").strip()
        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("Encerrando.")
            break

        resposta = qa_chain({"query": pergunta})
        print("Resposta:", resposta["result"])

if __name__ == "__main__":
    main()
