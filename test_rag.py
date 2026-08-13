from rag import RAG

rag = RAG()

question = "What is an authoritative restore of SYSVOL?"

result = rag.ask(question)

print("=" * 80)
print("Question:")
print(question)

print("\n" + "=" * 80)
print("Answer:")
print(result["answer"])

print("\n" + "=" * 80)
print("Sources:")

for src in result["sources"]:
    print(f"- {src['source']} (Page {src['page']})")