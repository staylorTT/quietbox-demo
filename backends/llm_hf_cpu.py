from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class ResponderHFCPU:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_new_tokens=128):
        print(f"[DEBUG] Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"[DEBUG] Tokenizer loaded. Loading model (this will download ~2.3GB on first run)...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"[DEBUG] Model loaded. Creating pipeline...")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=-1)
        print(f"[DEBUG] Pipeline created.")
        self.max_new_tokens = max_new_tokens

    def respond(self, user_text):
        prompt = f"<|system|>You are a concise helpful developer assistant.<|user|>{user_text}<|assistant|>"
        out = self.pipe(prompt, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=0.7)
        # take the assistant part
        generated = out[0]["generated_text"]
        # crude split, enough for demo
        if "<|assistant|>" in generated:
            reply = generated.split("<|assistant|>")[-1].strip()
        else:
            reply = generated.strip()
        return reply
