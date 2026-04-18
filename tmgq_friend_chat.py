import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tmgq_packer import QuantizedLinear

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    import functools
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def load_packed_huggingface(model_name, ckpt_path):
    print(f"Fetching skeletal architecture for {model_name}...")
    # Downloads metadata only (KB), NOT the massive FP16 weights
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Building blank memory allocation matrix...")
    with torch.device('meta' if torch.cuda.is_available() else 'cpu'):
        model = AutoModelForCausalLM.from_config(config)

    print(f"Loading TMG-Q Ultra Packed Payload [{ckpt_path}]...")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    
    # Prefix un-wrapper (if applicable from DDP/Trainer)
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
            
    quantized_keys = set([k.rpartition('.')[0] for k in state_dict.keys() if 'qweight' in k])
    if quantized_keys:
        print(f"🔥 Detected {len(quantized_keys)} INT32 Pack-Boxes! Restructuring network dynamically...")
        for n in quantized_keys:
            m = rgetattr(model, n, None)
            if m is not None and isinstance(m, nn.Linear):
                qlayer = QuantizedLinear(m.in_features, m.out_features, bias=m.bias is not None, gs=128)
                
                # Pre-size to absorb the packed data
                qlayer.qweight = torch.empty(state_dict[f"{n}.qweight"].shape, dtype=torch.int32)
                qlayer.scales = torch.empty(state_dict[f"{n}.scales"].shape, dtype=torch.float16)
                qlayer.zeros = torch.empty(state_dict[f"{n}.zeros"].shape, dtype=torch.float16)
                qlayer.w_shape = torch.empty(state_dict[f"{n}.w_shape"].shape, dtype=torch.int32)
                
                pre, _, post = n.rpartition('.')
                parent = rgetattr(model, pre) if pre else model
                setattr(parent, post, qlayer)
                
    # Bypass strict dimension enforcement on empty buffers
    model.to_empty(device="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(state_dict, strict=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print("Neural Link Established!")
    return model, tokenizer, device

def chat_interface():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ckpt_path = "TinyLlama_3bit_TMGQ.pt"
    
    try:
        model, tokenizer, device = load_packed_huggingface(model_name, ckpt_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find {ckpt_path}! Make sure the exported model is in this directory.")
        return
        
    print("\n" + "="*50)
    print("💬 TMG-Q Ultra Native Terminal Chat")
    print("Type 'exit' or 'quit' to end the session.")
    print("="*50 + "\n")
    
    chat_history = []
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            chat_history.append({"role": "user", "content": user_input})
            
            # Apply TinyLlama chat template format
            prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Streaming generation simulation in console
            print("Bot:", end=" ", flush=True)
            
            # Note: For simplicity we generate all at once here, you can add TextStreamer later
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(response + "\n")
            
            chat_history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    chat_interface()
