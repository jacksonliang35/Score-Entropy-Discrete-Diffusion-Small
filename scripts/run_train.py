import os
import yaml
# import oxen
import datetime
import gc
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import argparse

# from sedd.datasets.ox_dataset import OxDataset
# from sedd.datasets.brown_cow_dataset import BrownCowDataset
# from sedd.datasets.wikitext2_dataset import Wikitext2Dataset
# from sedd.datasets.open_subtitles_dataset import OpenSubtitlesDataset
# from sedd.datasets.baby_names_dataset import BabyNamesDataset
# from sedd.datasets.abc_dataset import ABCDataset

from sedd.datasets.text8_dataset import make_text8_loaders

# from sedd.tokenizers.abc_tokenizer import ABCTokenizer
from sedd.models.sedd import SEDD
from sedd.models.sampler import Sampler
from sedd.models.graph import UniformGraph, AbsorbingGraph
from sedd.models.noise import GeometricNoise, LogLinearNoise
from sedd.trainer.trainer import Trainer
from sedd.eval.evaluator import Evaluator
# from transformers import GPT2TokenizerFast

from aim import Run

# from sedd.models.simple_sedd import SEDD
from torch.utils.data import DataLoader

def print_devices(device):
    if torch.cuda.is_available():
        print("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        print("WARNING: Using device {}".format(device))
    print(f"Using device: {device}")
    print(f"Found {os.cpu_count()} total number of CPUs.")

def main():
    args = argparse.ArgumentParser(description="Train SEDD")
    args.add_argument("--cfg", type=str, default="configs/config.yaml")
    args.add_argument("--output", type=str, default="output")
    # args.add_argument("--repo", type=str, default="ox/SEDD_dev")
    args = args.parse_args()

    # load in tokenizer
    # tokenizer = OxTokenizer()
    # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    # print("Got EOS token: ", tokenizer.eos_token)
    # tokenizer.pad_token = '' # make sure we pad with absorbing token

    with open(args.cfg, 'r') as f:
        cfg = yaml.full_load(f)

    # cfg['data'] = {}
    # cfg['data']['remote_repo'] = args.repo
    cfg['training']['output_dir'] = args.output

    # Create directories for experimental logs
    work_dir = cfg['training']['output_dir']
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    data_dir = os.path.join(work_dir, "data")
    # checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    # os.path.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    print(work_dir)
    print(cfg)

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print_devices(device)

    # # logging
    # logger = utils.get_logger(os.path.join(work_dir, "logs"))
    # def mprint(msg):
    #     if rank == 0:
    #         logger.info(msg)
    #
    # mprint(work_dir)
    # mprint(cfg)
    # device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # if device.type == "cuda":
    #     mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
    #     for i in range(torch.cuda.device_count()):
    #         props = torch.cuda.get_device_properties(i)
    #         mprint(
    #             "{} \t Memory: {:.2f}GB".format(
    #                 props.name, props.total_memory / (1024 ** 3)
    #             )
    #         )
    # else:
    #     mprint("WARNING: Using device {}".format(device))
    # mprint(f"Found {os.cpu_count()} total number of CPUs.")

    # # Create remote oxen repo
    # repo = oxen.RemoteRepo(cfg['data']['remote_repo'])
    # if not repo.exists():
    #     repo.create()

    # # Save config file for this run
    # repo.add('configs/config.yaml')
    # repo.commit("Added config file")

    # train_ds = DataLoader(OpenSubtitlesDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=10_000), batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    # eval_ds = DataLoader(OpenSubtitlesDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=128))

    # train_ds = DataLoader(BabyNamesDataset(tokenizer, seq_len=cfg['model']['length']), batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    # eval_ds = DataLoader(BabyNamesDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=128, train=False))

    # train_ds = DataLoader(ABCDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=10000), batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    # eval_ds = DataLoader(ABCDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=128))

    train_ds, vocab = make_text8_loaders(
        data_dir,
        block_size=cfg['model']['length'],
        batch_size=cfg['training']['batch_size'],
        num_examples = 32768
    )
    eval_ds, _ = make_text8_loaders(
        data_dir,
        block_size=cfg['model']['length'],
        batch_size=cfg['eval']['batch_size'],
        train=False,
        num_examples = 128
    )
    cfg['tokens'] = vocab.size
    print("Number of tokens:", cfg['tokens'])

    # build model
    graph = UniformGraph(cfg['tokens'])
    score_model = SEDD(cfg, cfg['tokens']).to(device)
    noise = GeometricNoise(sigma_min=float(cfg['noise']['sigma_min']), sigma_max=float(cfg['noise']['sigma_max'])).to(device)
    num_parameters = sum(p.numel() for p in score_model.parameters())
    print(f"Number of parameters in the model: {num_parameters}")

    run = Run(experiment="sedd-char")
    run["hparams"] = cfg

    def eval(state):
        evaluator = Evaluator(eval_ds, cfg, run=run, device=device)
        return evaluator.evaluate(state)

    def sample(state):
        step = state['step']
        model = state['model']
        graph = state['graph']
        noise = state['noise']

        sampler = Sampler(cfg)
        texts = sampler.sample(vocab, model, graph, noise, steps=128, batch_size=cfg['eval']['batch_size'])

        file_name = os.path.join(sample_dir, f"sample.txt")
        with open(file_name, 'w') as file:
            for sentence in texts:
                file.write(sentence + "\n")
                file.write("="*80 + "\n")

        # # Push samples to Oxen.ai for tracking
        # repo = oxen.RemoteRepo(cfg['data']['remote_repo'])
        # repo.add(file_name)
        # repo.commit(f"Sample at step {step}")

    trainer = Trainer(
        run,
        score_model,
        graph,
        noise,
        cfg,
        eval_callback=eval,
        sample_callback=sample,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    trainer.train(train_ds)


if __name__ == "__main__":
    main()
