from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# load pretrained BERT embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def build_similarity(data):

    # convert course text into embeddings
    embeddings = model.encode(data["content"].tolist(), show_progress_bar=True)

    # compute cosine similarity
    similarity = cosine_similarity(embeddings)

    return similarity