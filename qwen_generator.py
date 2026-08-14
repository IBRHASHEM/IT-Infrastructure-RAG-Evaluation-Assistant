from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL_PATH = r"D:\Models\Qwen2.5-3B-Instruct"


class QwenGenerator:

    def __init__(self):

        print("Loading local Qwen model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )

        self.model.eval()

        print("Qwen model loaded successfully.")

    def generate(self, question, context):

        prompt = f"""You are an IT Infrastructure RAG assistant.

STRICT RULES:

1. Answer ONLY from the provided Context.
2. Do NOT use your own knowledge.
3. Do NOT add information that is not explicitly stated in the Context.
4. If the Context does not contain the answer, respond exactly:
I don't know based on the indexed documentation.
5. Keep the answer concise.
6. Do not invent examples, features, technologies, or details.

Context:
{context}

Question:
{question}

Answer:
"""

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return answer.strip()