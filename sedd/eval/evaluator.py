
from tqdm import tqdm
import math
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from ..trainer.loss import step_fn

class Evaluator:
    def __init__(self, dataset, cfg, run=None, device = 'cuda'):
        self.dataset = dataset
        self.run = run
        self.cfg = cfg
        self.device = device

    def evaluate(self, state):
        step = state['step']
        sum_loss = 0
        print(f"Evaluating model on validation set")
        for batch in tqdm(self.dataset):
            batch = batch.to(self.device)
            loss = self.evaluate_batch(state, batch)
            sum_loss += loss.item()
        avg_loss = sum_loss / len(self.dataset)
        print("step: %d, evaluation_loss: %.5e" % (step, avg_loss))
        if self.run is not None:
            self.run.track(avg_loss, name='loss', step=state['step'], context={ "subset":"eval" })
        return avg_loss

    def evaluate_batch(self, state, batch):
        model = state['model']
        model.eval()
        with torch.no_grad():
            eval_loss = step_fn(self.cfg, state, batch, train=False)
            return eval_loss

    @torch.no_grad()
    def evaluate_ppl(text, block_size=32):
        """
        text: len(N) text
        """
        device = self.device
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.to(device)
        model.eval()

        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        input_ids = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]

        nll = 0.0
        n_tokens = 0

        for i in range(0, input_ids.size(1) - 1, block_size):
            x = input_ids[:, i : i + block_size].to(device)
            y = input_ids[:, i + 1 : i + block_size + 1].to(device)

            if y.numel() == 0:
                break

            outputs = model(x)
            logits = outputs.logits  # [1, T, V]

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction="sum",
            )

            nll += loss.item()
            n_tokens += y.numel()

        avg_nll = nll / n_tokens
        ppl = math.exp(avg_nll)
        return ppl
