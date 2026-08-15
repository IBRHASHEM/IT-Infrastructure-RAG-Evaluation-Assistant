
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_NAME


class QwenGenerator:

    def __init__(self):

        print("Loading local Qwen model...")
        print(f"Model path: {MODEL_NAME}")

        # -------------------------------------------------
        # Tokenizer
        # -------------------------------------------------

        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

        print("Tokenizer loaded.")

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        print("Loading model weights...")

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
            dtype=torch.float32,            
            low_cpu_mem_usage=True,
        )

        print("Model weights loaded.")

        self.model.eval()

        if self.tokenizer.pad_token_id is None:

            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
            )

        print("Qwen model loaded successfully.")

    # -----------------------------------------------------
    # Generation
    # -----------------------------------------------------

    def generate(self, question, context):

        prompt = f"""You are an IT documentation assistant.

Answer ONLY from the documentation.

Do not use outside knowledge.
Do not guess.
Do not invent information.

If the documentation does not contain the answer, respond exactly:

I don't know based on the indexed documentation.

DOCUMENTATION:
{context}

QUESTION:
{question}

ANSWER:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072,
        )

        input_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                repetition_penalty=1.10,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0][input_length:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        if not answer:

            return (
                "I don't know based on the indexed documentation."
            )

        return answer
