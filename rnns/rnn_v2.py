
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


EMBEDDING_DIM = 64
HIDDEN_DIM = 256
SEQ_LEN = 64
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3


class LMDataset(Dataset):
    """Each item is a window of tokens and that same window shifted by one,
    so the model predicts token t+1 from tokens 0..t at every position."""

    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, i):
        return (self.data[i : i + self.seq_len],
                self.data[i + 1 : i + self.seq_len + 1])


class TextGenerator(nn.Module):
    def __init__(self, dict_size):
        super(TextGenerator, self).__init__()

        self.embedding = nn.Embedding(dict_size, EMBEDDING_DIM)
        self.rnn = nn.RNN(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_DIM,
                          num_layers=1, batch_first=True)
        self.head = nn.Linear(HIDDEN_DIM, dict_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)                 # (B, T) -> (B, T, E)
        out, hidden = self.rnn(x, hidden)     # (B, T, H)
        return self.head(out), hidden         # (B, T, V)


@torch.no_grad()
def generate(model, vocab, word_map, prompt, n_words=30, temperature=1.0):
    model.eval()

    words = prompt.split()
    ids = [word_map[w] for w in words if w in word_map]
    if not ids:
        ids = [0]

    hidden = None
    x = torch.tensor(ids).unsqueeze(0)        # (1, T)

    for _ in range(n_words):
        logits, hidden = model(x, hidden)
        logits = logits[0, -1] / temperature   # last timestep only
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        words.append(vocab[next_id.item()])
        x = next_id.unsqueeze(0)               # feed just the new token

    model.train()
    return " ".join(words)


def main():
    with open("rnns/input.txt", mode='r') as file:
        tokens = file.read().split()

    vocab = sorted(set(tokens))
    word_map = {word: i for i, word in enumerate(vocab)}
    dict_size = len(vocab)
    print(f"{len(tokens)} tokens, vocab size {dict_size}")

    data = torch.tensor([word_map[w] for w in tokens], dtype=torch.long)

    loader = DataLoader(LMDataset(data, SEQ_LEN), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    model = TextGenerator(dict_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, dict_size), y.reshape(-1))
            loss.backward()

            # vanilla RNNs blow up without this
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        avg = running_loss / len(loader)
        print(f"epoch {epoch} done | avg loss {avg:.4f}")

        model_cpu = model.to("cpu")
        print(generate(model_cpu, vocab, word_map, prompt=tokens[0]))
        model.to(device)

    torch.save({"state_dict": model.state_dict(), "vocab": vocab},
               "rnns/text_generator.pt")


if __name__ == "__main__":
    main()
