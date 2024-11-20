import os

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"

from rwkv_model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

model = RWKV(model="sudoku_rwkv_20241120.pth", strategy="cuda fp16", verbose=False)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
pipeline.tokenizer = TRIE_TOKENIZER("sudoku_vocab.txt")
gen_args = PIPELINE_ARGS(top_k=1, alpha_frequency=0, alpha_presence=0, token_stop=[105])

# make sure your sudoku has exactly one solution. launch.py will verify this automatically.
input_str = '''<input>
0 0 8 1 6 7 0 2 0 
5 0 0 2 3 0 0 0 0 
7 6 0 0 5 4 8 0 1 
8 7 0 0 4 0 0 0 0 
0 2 0 0 0 0 0 0 0 
0 0 4 0 0 3 0 9 0 
0 0 0 0 0 0 3 7 0 
0 4 0 0 0 0 0 8 0 
3 1 0 8 0 6 9 0 4 
</input>

'''

print(f'{" Model input ":-^100}\n{input_str}\n{" Model output ":-^100}')
pipeline.generate(input_str, token_count=50000, args=gen_args, callback=lambda x: print(x, end="", flush=True))