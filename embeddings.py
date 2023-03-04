import numpy as np
import openai
import pandas as pd

EMBEDDING_MODEL = "text-embedding-ada-002"


def read_csv(file: str, header: int) -> pd.DataFrame:
    return pd.read_csv(file, header=header)


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]


def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }


def save_embeddings(embeddings: dict[tuple[str, str], list[float]], fname: str):
    title_list, description_list, embedding_list = [], [], []
    for k, v in embeddings.items():
        title_list.append(k[0])
        description_list.append(k[1])
        embedding_list.append(v)

    df_dict = {'title': title_list, 'description': description_list}

    for i in range(len(embedding_list[0])):
        df_dict[str(i)] = [e[i] for e in embedding_list]

    df = pd.DataFrame(df_dict)
    df.set_index(['title', 'description'], inplace=True)
    df.to_csv(fname, index=True, header=True)


def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    df = read_csv(fname, 0)
    max_dim = max([int(c) for c in df.columns if c != 'title' and c != 'description'])
    return {
        (r.title, r.description): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }


def vector_similarity(x: list[float], y: list[float]) -> float:
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[
    (float, (str, str))]:
    query_embedding = get_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities
