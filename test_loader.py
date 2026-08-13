from document_loader import DocumentLoader

loader = DocumentLoader("data")

documents = loader.load()

print(f"\nTotal pages with text: {len(documents)}")

if documents:
    print("\nFirst page:")
    print(f"Source : {documents[0]['source']}")
    print(f"Page   : {documents[0]['page']}")
    print(documents[0]["text"][:500])