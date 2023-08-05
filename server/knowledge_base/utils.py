import os
from configs.model_config import KB_ROOT_PATH
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from configs.model_config import (CACHED_VS_NUM, VECTOR_SEARCH_TOP_K,
                                  embedding_model_dict, EMBEDDING_MODEL, EMBEDDING_DEVICE)
from functools import lru_cache


_VECTOR_STORE_TICKS = {}


def get_kb_path(knowledge_base_name: str):
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)


def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")


def get_vs_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "vector_store")


def get_file_path(knowledge_base_name: str, doc_name: str):
    return os.path.join(get_doc_path(knowledge_base_name), doc_name)


def validate_kb_name(knowledge_base_id: str) -> bool:
    # 检查是否包含预期外的字符或路径攻击关键字
    if "../" in knowledge_base_id:
        return False
    return True


@lru_cache(1)
def load_embeddings(model: str, device: str):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[model],
                                       model_kwargs={'device': device})
    return embeddings


@lru_cache(CACHED_VS_NUM)
def load_vector_store(
        knowledge_base_name: str,
        embedding_model: str,
        embedding_device: str,
        tick: int,  # tick will be changed by upload_doc etc. and make cache refreshed.
):
    print(f"loading vector store in '{knowledge_base_name}' with '{embedding_model}' embeddings.")
    embeddings = load_embeddings(embedding_model, embedding_device)
    vs_path = get_vs_path(knowledge_base_name)
    search_index = FAISS.load_local(vs_path, embeddings)
    return search_index


def lookup_vs(
        query: str,
        knowledge_base_name: str,
        top_k: int = VECTOR_SEARCH_TOP_K,
        embedding_model: str = EMBEDDING_MODEL,
        embedding_device: str = EMBEDDING_DEVICE,
):
    search_index = load_vector_store(knowledge_base_name,
                                     embedding_model,
                                     embedding_device,
                                     _VECTOR_STORE_TICKS.get(knowledge_base_name))
    docs = search_index.similarity_search(query, k=top_k)
    return docs


def refresh_vs_cache(kb_name: str):
    """
    make vector store cache refreshed when next loading
    """
    _VECTOR_STORE_TICKS[kb_name] = _VECTOR_STORE_TICKS.get(kb_name, 0) + 1
