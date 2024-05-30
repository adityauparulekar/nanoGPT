"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import numpy as np
import sys
import tiktoken
from model import GPTConfig, GPT
from collections import defaultdict
from torch.nn import functional as F



# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
bias = 0.5
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1000 # number of samples to draw
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 1 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile_bool = False # use PyTorch 2.0 to compile the model to be faster
num=0
p=0.0
iter=0
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
print("HELLO", out_dir, bias, num, p, iter)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
print(out_dir)
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile_bool:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)
# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    print(stoi, itos)
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

def check_syntax(filename):
    try:
        with open(filename, 'r') as file:
            code = file.read()
            compile(code, filename, 'exec')
        return True
    except SyntaxError as e:
        return False


# run generation
markov = {}
size = 2
rng = np.random.default_rng(42)
for s in range(1, size+1):
    # print(s)
    for i in range(2**s):
        curr_seed = format(i, f'0{s}b')
        # print(i, curr_seed)
        bias = rng.choice([0.25, 0.5, 0.75])
        markov[curr_seed] = bias
markov[''] = 0.5

def category(f):
    is01 = True
    is23 = True
    for i in range(1, len(f)):
        c = f[i]
        if c not in '01':
            is01 = False
        if c not in '23':
            is23 = False
    if is01:
        return 0
    elif is23:
        return 1
    return -1
    
prob_diff_array = []
oz = 0
tt = 0
emp_markov = defaultdict(float)
with torch.no_grad():
    with ctx:
        for k in range(10):
            y = model.generate_probs(x, max_new_tokens, temperature=temperature, top_k=None)
            final_out = decode(y[0][0].tolist())
            print(final_out)
            cat = category(final_out)
            prob_diffs = []
            if cat == -1:
                continue
            if cat == 0:
                oz += 1
                probs = y[1]
                for i in range(len(probs)):
                    context = final_out[1:i+1][-size:]
                    zero_p = probs[i][1] / (probs[i][1] + probs[i][2])
                    prob_diffs.append(abs(zero_p - markov[context]))
                    emp_markov[context] = zero_p
            else:
                tt += 1
            prob_diff_array.append(np.mean(prob_diffs))
prob_diff_array = np.array(prob_diff_array)
print(prob_diff_array)
# l = []
# for k in emp_markov:
#     l.append(abs(markov[k] - emp_markov[k]))
# f = open('rare_history_outputs_a.txt', 'a')
# if l:
#     f.write(f"num:{num}, p:{p}, meanerror:{np.mean(l)}\n")
#     f.write(str(emp_markov) + '\n')
# else:
#     f.write(f"num:{num}, p:{p}, meanerror:-1\n")
# f.close()

k = markov.keys()
for c in k:
    start_ids = encode(c)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    logits, _ = model(x)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    print(c, (probs[0][1]/(probs[0][1] + probs[0][2])).item(), markov[c])
print(markov)

# {'0': 0.25, '1': 0.75, '00': 0.5, '01': 0.5, '10': 0.5, '11': 0.75, '000': 0.25, '001': 0.75, '010': 0.25, '011': 0.25, '100': 0.5, '101': 0.75, '110': 0.75, '111': 0.75}