from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:

    def __init__(self):
        print("Loading BGE model...")
        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

    def embed(self, text):

        return self.model.encode(
            text,
            normalize_embeddings=True
        ).tolist()

    def embed_batch(self, texts):

        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()



#============================================
# # import time

# from google import genai
# from google.genai.errors import ServerError, ClientError

# from config import GOOGLE_API_KEY, EMBEDDING_MODEL


# class EmbeddingGenerator:

#     def __init__(self):
#         self.client = genai.Client(api_key=GOOGLE_API_KEY)

#     def embed(self, text: str):
#         response = self.client.models.embed_content(
#             model=EMBEDDING_MODEL,
#             contents=text,
#         )

#         return response.embeddings[0].values

#     def embed_batch(self, texts):

#         print(f"Sending {len(texts)} texts to Gemini...")

#         embeddings = []

#         for i, text in enumerate(texts, start=1):

#             while True:

#                 try:

#                     response = self.client.models.embed_content(
#                         model=EMBEDDING_MODEL,
#                         contents=text,
#                     )

#                     embeddings.append(response.embeddings[0].values)

#                     print(f"  {i}/{len(texts)}", end="\r")

#                     break

#                 except ServerError:

#                     print("\nGemini unavailable. Waiting 10 seconds...")
#                     time.sleep(10)

#                 except ClientError as e:

#                     if e.status_code == 429:
#                         print("\nRate limit reached. Waiting 20 seconds...")
#                         time.sleep(20)
#                     else:
#                         raise

#         print()

#         return embeddings