from typing import List, Dict


class TextChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents: List[Dict]) -> List[Dict]:

        chunks = []

        for document in documents:

            text = document["text"]

            start = 0

            while start < len(text):

                end = start + self.chunk_size

                chunk = text[start:end]

                chunks.append(
                    {
                        "id": f'{document["source"]}-{document["page"]}-{len(chunks)}',
                        "text": chunk,
                        "source": document["source"],
                        "page": document["page"],
                    }
                )

                start += self.chunk_size - self.overlap

        return chunks