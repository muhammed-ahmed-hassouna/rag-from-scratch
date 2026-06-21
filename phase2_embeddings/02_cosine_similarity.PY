from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Three sentences: two similar, one unrelated
s1 = "How do I reset my password?"
s2 = "I forgot my password, how can I recover it?"
s3 = "What is the best recipe for mansaf?"

e1 = model.encode(s1)
e2 = model.encode(s2)
e3 = model.encode(s3)

print(f"s1 vs s2 (should be HIGH): {cosine_similarity(e1, e2):.4f}")
print(f"s1 vs s3 (should be LOW):  {cosine_similarity(e1, e3):.4f}")

def Printer(name, embedding):
    print(f"{name}: {embedding}")

Printer("e1", e1[:5])
Printer("e2", e2[:5])
Printer("e3", e3[:5])