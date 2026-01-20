import os
import torch
import argparse
import yaml
# from transformers import GPT2TokenizerFast
from sedd.tokenizers.abc_tokenizer import ABCTokenizer

from sedd.models.sedd import SEDD
from sedd.models.graph import AbsorbingGraph
from sedd.models.noise import LogLinearNoise
from sedd.models.sampler import Sampler

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--cfg", type=str, default="configs/config.yaml")
    # parser.add_argument("--tokenizer", default="gpt2", type=str)
    parser.add_argument("--show_intermediate", action='store_true')
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()

    # Config should be saved in the model directory
    with open(args.cfg, 'r') as f:
        cfg = yaml.full_load(f)

    # Load the tokenizer
    # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer = ABCTokenizer()

    print("Vocab size: ", tokenizer.vocab_size)
    print("Last token in vocab: ", tokenizer.decode([tokenizer.vocab_size-1]))
    print("Past vocab size: ", tokenizer.batch_decode([[tokenizer.vocab_size]]))

    # Load the model onto GPU
    device = torch.device('cuda')
    loaded_state = torch.load(
        os.path.join(self.checkpoint_dir, "checkpoint.pth"),
        map_location=self.device
    )
    model = SEDD(cfg, tokenizer.vocab_size).to(device)
    model.load_state_dict(loaded_state['model'])

    # Load the transition graph
    graph = AbsorbingGraph(tokenizer.vocab_size)

    # Load the noise function
    noise = LogLinearNoise().to(device)

    sampler = Sampler(cfg, device=device)
    texts = sampler.sample(tokenizer, model, graph, noise, steps=args.steps, show_intermediate=args.show_intermediate)

    for i in texts:
        print("="*80)
        print(i)
        print("="*80)

if __name__=="__main__":
    main()
