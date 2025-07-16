from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


class ChunkRetriever:
    def __init__(self, vector_db_path, meta_path, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(vector_db_path)
        self.chunk_texts = np.load(meta_path, allow_pickle=True)

    def retrieve(self, query, top_k=4):
        q_emb = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            results.append(
                {'chunk': self.chunk_texts[idx], 'score': float(score), 'idx': int(idx)})
        return results
