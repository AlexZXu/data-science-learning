"""Visual runner for the trained TextGenerator: streams one token at a time.

    python3 rnns/run_model.py --prompt "first citizen" --n-words 60
"""

import argparse
import sys
import time

import torch

from rnn_v2 import TextGenerator

CHECKPOINT = "rnns/text_generator.pt"


@torch.no_grad()
def stream(model, vocab, word_map, prompt, n_words, temperature, delay):
    given = prompt.split()
    unknown = [w for w in given if w not in word_map]
    if unknown:
        print(f"(not in vocab, dropped: {' '.join(unknown)})\n", file=sys.stderr)

    # only the kept words are printed, so what you see is what the model saw
    words = [w for w in given if w in word_map]
    ids = [word_map[w] for w in words]
    if not ids:
        ids = [0]
        words = [vocab[0]]
        print("(no usable prompt tokens, starting from vocab[0])\n",
              file=sys.stderr)

    # print the prompt itself token by token so the whole run looks uniform
    for w in words:
        sys.stdout.write(w + " ")
        sys.stdout.flush()
        time.sleep(delay)

    hidden = None
    x = torch.tensor(ids).unsqueeze(0)        # (1, T)

    for _ in range(n_words):
        logits, hidden = model(x, hidden)
        logits = logits[0, -1] / temperature   # last timestep only
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)

        sys.stdout.write(vocab[next_id.item()] + " ")
        sys.stdout.flush()
        time.sleep(delay)

        x = next_id.unsqueeze(0)               # feed just the new token

    print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Good")
    parser.add_argument("--n-words", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--delay", type=float, default=0.08,
                        help="seconds between tokens (0 for instant)")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    word_map = {word: i for i, word in enumerate(vocab)}

    model = TextGenerator(len(vocab))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"vocab size {len(vocab)} | temperature {args.temperature}\n")
    stream(model, vocab, word_map, args.prompt, args.n_words,
           args.temperature, args.delay)


if __name__ == "__main__":
    main()
