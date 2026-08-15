from gemini_generator import GeminiGenerator


print("=" * 70)
print("GEMINI GENERATOR TEST")
print("=" * 70)


generator = GeminiGenerator()


context = """
VMware vMotion allows a running virtual machine to be migrated
from one ESXi host to another without downtime.
"""


question = "What is VMware vMotion?"


print()
print("Generating answer...")
print()


answer = generator.generate(
    question=question,
    context=context
)


print("=" * 70)
print("ANSWER")
print("=" * 70)

print(answer)

print("=" * 70)
print("TEST COMPLETED")
print("=" * 70)