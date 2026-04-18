import sys
from pathlib import Path
import torch
import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DATASET_DIR = ROOT / "data" / "self_code_char"
sys.path.insert(0, str(DATASET_DIR))

try:
    from model import GPT, GPTConfig
except ImportError:
    print("Error: Could not import nanoGPT model.")
    sys.exit(1)

import pickle

def load_meta():
    with open(DATASET_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return meta["stoi"], meta["itos"]

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = "out-self-code-long/ckpt_3bit.pt" # Correct path to the 3-bit long trained model

# Load model
print(f"Loading {ckpt_path}...")
try:
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    stoi, itos = load_meta()
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    print("Done loading model!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

quant_bits = checkpoint.get("omegaquant", {}).get("bits", "Unknown")

def generate_text(prompt, max_new_tokens=200, temperature=0.8, top_k=200):
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    return decode(y[0].tolist())

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown(f"# TMG-Q Ultra : Live {quant_bits}-bit Inference Demo")
    gr.Markdown("Interact directly with your fully autonomous NanoGPT model running locally.")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, type="messages")
            msg = gr.Textbox(label="Enter Prompt (Python Context)")
        with gr.Column(scale=1):
            temp_slider = gr.Slider(0.1, 2.0, value=0.8, label="Temperature")
            tokens_slider = gr.Slider(10, 500, value=200, step=10, label="Max Tokens")
            clear = gr.ClearButton([msg, chatbot])

    def respond(user_message, chat_history, temp, max_tokens):
        bot_message = generate_text(user_message, max_tokens, temp)
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history

    msg.submit(respond, [msg, chatbot, temp_slider, tokens_slider], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
