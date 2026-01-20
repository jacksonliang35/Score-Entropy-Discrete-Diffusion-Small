import os
import torch
import argparse
import yaml
# from sedd.tokenizers.abc_tokenizer import ABCTokenizer

from sedd.models.sedd import SEDD
from sedd.models.graph import UniformGraph, AbsorbingGraph
from sedd.models.noise import GeometricNoise, LogLinearNoise
from sedd.models.sampler import Sampler
from sedd.eval.evaluator import Evaluator

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

    # Load the evaluation data
    eval_ds, vocab = make_text8_loaders(
        block_size=cfg['model']['length'],
        batch_size=cfg['eval']['batch_size'],
        train=False,
        num_examples=128
    )

    device = torch.device('cuda')

    # print("Vocab size: ", tokenizer.vocab_size)
    # print("Last token in vocab: ", tokenizer.decode([tokenizer.vocab_size-1]))
    # print("Past vocab size: ", tokenizer.batch_decode([[tokenizer.vocab_size]]))

    # Load the model onto GPU

    loaded_state = torch.load(
        os.path.join(self.checkpoint_dir, "checkpoint.pth"),
        map_location=self.device
    )
    vocab_size = cfg['tokens'].size
    sedd_model = SEDD(cfg, vocab_size).to(device)
    sedd_model.load_state_dict(loaded_state['model'])

    # Load the transition graph
    graph = UniformGraph(vocab_size)

    # Load the noise function
    noise = GeometricNoise(sigma_min=cfg['noise']['sigma_min'], sigma_max=cfg['noise']['sigma_max']).to(device)

    sampler = Sampler(cfg, device=device)
    texts = sampler.sample(vocab, sedd_model, graph, noise, steps=args.steps, show_intermediate=args.show_intermediate)

    for i in texts:
        print("="*80)
        print(i)
        print("="*80)

    Evaluator(eval_ds, cfg, device=device).evaluate_ppl(texts)

if __name__=="__main__":
    main()
