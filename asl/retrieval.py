"""Cosine-prototype few-shot sign retrieval (no trained features supplied)."""
import numpy as np

class PrototypeIndex:
    def __init__(self):
        self.labels=[]
        self.prototypes=None

    @staticmethod
    def normalize(embeddings):
        arr=np.asarray(embeddings,dtype=np.float32)
        if arr.ndim!=2 or not np.isfinite(arr).all() or len(arr)==0:
            raise ValueError("Expected finite embeddings with shape (N,D)")
        return arr / np.maximum(np.linalg.norm(arr,axis=1,keepdims=True),1e-12)

    def fit(self,embeddings,labels):
        emb=self.normalize(embeddings)
        y=np.asarray(labels)
        if len(y)!=len(emb):
            raise ValueError("Labels and embeddings must align")
        self.labels=sorted({str(v) for v in y})
        if any(not v for v in self.labels):
            raise ValueError("Empty class label")
        proto=np.stack([emb[np.array([str(v)==label for v in y])].mean(axis=0) for label in self.labels])
        self.prototypes=self.normalize(proto)
        return self

    def query(self,embeddings,k=5):
        if self.prototypes is None:
            raise RuntimeError("Index must be fitted")
        q=self.normalize(embeddings)
        if q.shape[1]!=self.prototypes.shape[1] or k<1:
            raise ValueError("Feature dimension mismatch or invalid k")
        similarities=q@self.prototypes.T
        indices=np.argsort(-similarities,axis=1)[:,:min(k,len(self.labels))]
        return [[{"label":self.labels[j],"similarity":float(similarities[i,j])} for j in row] for i,row in enumerate(indices)]
