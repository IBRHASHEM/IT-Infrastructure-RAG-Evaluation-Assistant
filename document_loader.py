from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


class DocumentLoader:
    """
    Load all PDF documents from a directory.
    """

    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)

    def load(self) -> List[Dict]:

        documents = []

        pdf_files = sorted(self.data_path.glob("*.pdf"))

        print(f"Found {len(pdf_files)} PDF file(s).")

        for pdf_file in pdf_files:

            print(f"\nLoading {pdf_file.name}")

            try:
                reader = PdfReader(pdf_file)

                print(f"Pages: {len(reader.pages)}")

                extracted = 0

                for page_number, page in enumerate(reader.pages, start=1):

                    text = page.extract_text()

                    if not text:
                        continue

                    documents.append(
                        {
                            "text": text,
                            "source": pdf_file.name,
                            "page": page_number,
                        }
                    )

                    extracted += 1

                print(f"Extracted {extracted} pages")

            except Exception as e:
                print(f"Error reading {pdf_file.name}: {e}")

        print(f"\nLoaded {len(documents)} pages with text.")

        return documents