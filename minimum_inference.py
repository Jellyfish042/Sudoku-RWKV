import os

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"

from rwkv_model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

model = RWKV(model="sudoku_rwkv_20241029.pth", strategy="cuda fp16", verbose=False)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
pipeline.tokenizer = TRIE_TOKENIZER("sudoku_vocab_v6.txt")
gen_args = PIPELINE_ARGS(top_k=1, alpha_frequency=0, alpha_presence=0, token_stop=[127])

# make sure your sudoku has exactly one solution. launch.py will verify this automatically.
input_str = '''<input>
[[0, 0, 0, 7, 1, 0, 0, 9, 0], \
[0, 1, 7, 0, 6, 9, 2, 8, 0], \
[3, 9, 0, 0, 8, 2, 0, 7, 0], \
[9, 7, 0, 2, 4, 6, 8, 5, 1], \
[6, 0, 8, 0, 9, 3, 7, 4, 0], \
[0, 0, 2, 5, 7, 8, 0, 0, 0], \
[8, 2, 9, 6, 5, 7, 4, 0, 0], \
[4, 0, 0, 8, 0, 1, 0, 2, 7], \
[7, 3, 1, 9, 0, 0, 5, 6, 8]]'''

print(f'{" Model input ":-^100}\n{input_str}\n{" Model output ":-^100}')
pipeline.generate(input_str, token_count=500000, args=gen_args, callback=lambda x: print(x, end="", flush=True))