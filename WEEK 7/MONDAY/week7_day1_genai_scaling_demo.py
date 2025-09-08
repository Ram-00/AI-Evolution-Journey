import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def benchmark_generation(model_name, prompt, device="cpu", runs=3):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tok.eos_token_id
    times, outputs = [], []
    for _ in range(runs):
        t0 = time.time()
        enc = tok(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=32,
                pad_token_id=model.config.pad_token_id,
            )
        dt = time.time() - t0
        txt = tok.batch_decode(out, skip_special_tokens=True)[0]
        times.append(dt); outputs.append(txt)
    return {"latency": times, "outputs": outputs}

if __name__ == "__main__":
    cloud_model = "gpt2"            # as cloud proxy
    edge_model  = "sshleifer/tiny-gpt2"   # lightweight edge-ready proxy for CPU
    prompt = "Summarize the latest generative AI scaling trends for enterprise in 2025:"
    cloud_res = benchmark_generation(cloud_model, prompt, device="cpu")
    edge_res = benchmark_generation(edge_model, prompt, device="cpu")
    print("Cloud model latencies (s):", cloud_res["latency"])
    print("Edge model latencies (s):", edge_res["latency"])
    print("\nSample output (cloud):", cloud_res["outputs"])
    print("\nSample output (edge):", edge_res["outputs"])
