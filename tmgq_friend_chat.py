import torch
import torch.nn as nn
from tmgq_transformers_compat import disable_broken_torchvision

disable_broken_torchvision()
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tmgq_packer import QuantizedEmbedding, QuantizedLinear, QuantizedTiedLMHead

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
    
    print("Building mathematical skeleton (allocating temporary CPU memory)...")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    print(f"Loading TMG-Q Ultra Packed Payload [{ckpt_path}]...")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    
    # Prefix un-wrapper (if applicable from DDP/Trainer)
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
            
    quantized_keys = set([k.rpartition('.')[0] for k in state_dict.keys() if 'qweight' in k])
    if quantized_keys:
        print(f"Detected {len(quantized_keys)} INT32 Pack-Boxes. Restructuring network dynamically...")
        embedding_keys = {n for n in quantized_keys if f"{n}.tied_lm_head" in state_dict}
        for n in embedding_keys:
            m = rgetattr(model, n, None)
            if m is None or not isinstance(m, nn.Embedding):
                continue
            bits = int(state_dict[f"{n}.n_bits"].item())
            group_size = int(state_dict[f"{n}.group_size"].item())
            qembedding = QuantizedEmbedding(
                m.num_embeddings,
                m.embedding_dim,
                gs=group_size,
                n_bits=bits,
            )
            qembedding.qweight = torch.empty(state_dict[f"{n}.qweight"].shape, dtype=torch.int32)
            qembedding.scales = torch.empty_like(state_dict[f"{n}.scales"])
            qembedding.zeros = torch.empty_like(state_dict[f"{n}.zeros"])
            qembedding.codebooks = torch.empty_like(state_dict[f"{n}.codebooks"])
            qembedding.w_shape = torch.empty(state_dict[f"{n}.w_shape"].shape, dtype=torch.int32)
            qembedding.pad_len = torch.empty(state_dict[f"{n}.pad_len"].shape, dtype=torch.int32)
            qembedding.n_bits = torch.tensor(bits, dtype=torch.int32)
            qembedding.group_size = torch.tensor(group_size, dtype=torch.int32)
            qembedding.tied_lm_head = torch.tensor(
                bool(state_dict[f"{n}.tied_lm_head"].item()),
                dtype=torch.bool,
            )
            if f"{n}.dtype_code" in state_dict:
                qembedding.dtype_code = torch.tensor(
                    int(state_dict[f"{n}.dtype_code"].item()),
                    dtype=torch.int32,
                )
            if f"{n}.svd_u" in state_dict:
                qembedding.svd_u = torch.empty_like(state_dict[f"{n}.svd_u"])
                qembedding.svd_s = torch.empty_like(state_dict[f"{n}.svd_s"])
                qembedding.svd_v = torch.empty_like(state_dict[f"{n}.svd_v"])
                if f"{n}.svd_u_scale" in state_dict:
                    qembedding.svd_u_scale = torch.empty_like(state_dict[f"{n}.svd_u_scale"])
                    qembedding.svd_v_scale = torch.empty_like(state_dict[f"{n}.svd_v_scale"])
            pre, _, post = n.rpartition(".")
            parent = rgetattr(model, pre) if pre else model
            setattr(parent, post, qembedding)
            if bool(qembedding.tied_lm_head.item()):
                model.set_output_embeddings(QuantizedTiedLMHead(qembedding))

        for n in quantized_keys - embedding_keys:
            m = rgetattr(model, n, None)
            if m is not None and (isinstance(m, nn.Linear) or m.__class__.__name__ == "Conv1D"):
                bits_key = f"{n}.n_bits"
                bits = int(state_dict[bits_key].item()) if bits_key in state_dict else 4
                group_key = f"{n}.group_size"
                if group_key in state_dict:
                    group_size = int(state_dict[group_key].item())
                else:
                    group_size = 128
                if isinstance(m, nn.Linear):
                    in_features, out_features = m.in_features, m.out_features
                else:
                    in_features, out_features = m.weight.shape
                qlayer = QuantizedLinear(in_features, out_features, bias=m.bias is not None, gs=group_size, n_bits=bits)
                
                # Pre-size to absorb the packed data
                qlayer.qweight = torch.empty(state_dict[f"{n}.qweight"].shape, dtype=torch.int32)
                qlayer.scales = torch.empty_like(state_dict[f"{n}.scales"])
                qlayer.zeros = torch.empty_like(state_dict[f"{n}.zeros"])
                if f"{n}.codebooks" in state_dict:
                    qlayer.codebooks = torch.empty_like(state_dict[f"{n}.codebooks"])
                qlayer.w_shape = torch.empty(state_dict[f"{n}.w_shape"].shape, dtype=torch.int32)
                qlayer.pad_len = torch.empty(state_dict[f"{n}.pad_len"].shape, dtype=torch.int32)
                qlayer.n_bits = torch.tensor(bits, dtype=torch.int32)
                qlayer.group_size = torch.tensor(group_size, dtype=torch.int32)
                if f"{n}.outlier_rows" in state_dict:
                    qlayer.outlier_rows = torch.empty(state_dict[f"{n}.outlier_rows"].shape, dtype=torch.int32)
                    qlayer.outlier_cols = torch.empty(state_dict[f"{n}.outlier_cols"].shape, dtype=torch.int32)
                    qlayer.outlier_values = torch.empty_like(state_dict[f"{n}.outlier_values"])
                if f"{n}.svd_u" in state_dict:
                    qlayer.svd_u = torch.empty_like(state_dict[f"{n}.svd_u"])
                    qlayer.svd_s = torch.empty_like(state_dict[f"{n}.svd_s"])
                    qlayer.svd_v = torch.empty_like(state_dict[f"{n}.svd_v"])
                
                pre, _, post = n.rpartition('.')
                parent = rgetattr(model, pre) if pre else model
                setattr(parent, post, qlayer)
    # Load the compressed INT32 dictionaries over the FP16 ones
    model.load_state_dict(state_dict, strict=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print("Neural Link Established!")
    return model, tokenizer, device

def chat_interface():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ckpt_path = "TinyLlama_4bit_TMGQ.pt"
    
    try:
        model, tokenizer, device = load_packed_huggingface(model_name, ckpt_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find {ckpt_path}! Make sure the exported model is in this directory.")
        return
        
    print("\n" + "="*50)
    print("TMG-Q Ultra Native Terminal Chat")
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
                    do_sample=False,
                    repetition_penalty=1.3,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(response + "\n")
            
            chat_history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    chat_interface()
