# class SimpleIndex:
#     def __init__(self):
#         self.index = []

#     def add(self, text, embedding):
#         self.index.append({"text": text, "embedding": embedding})

#     def search(self, query_embedding, top_k=1):
#         # Naive similarity (dot product)
#         sims = [
#             (item["text"], (query_embedding @ item["embedding"].T).item())
#             for item in self.index
#         ]
#         sims.sort(key=lambda x: x[1], reverse=True)
#         return sims[:top_k]
