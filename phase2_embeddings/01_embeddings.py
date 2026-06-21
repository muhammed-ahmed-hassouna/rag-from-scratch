from sentence_transformers import SentenceTransformer

# Load the model (downloads once ~90MB, then cached)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed a sentence
embedding = model.encode("Hello my name is Omar")

print(type(embedding))        # <class 'numpy.ndarray'>
print(embedding.shape)        # (384,)  → 384 numbers
print(embedding[:5])          # first 5 numbers, e.g. [ 0.023 -0.147  0.891  0.034 -0.223]
print(len(embedding))         # 384