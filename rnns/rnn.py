
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


EMBEDDING_DIM = 64

class TextGenerator(nn.Module):
    def __init__(self, dict_size):
        super(TextGenerator, self).__init__()

        self.rnn = nn.RNN(input_size=EMBEDDING_DIM, hidden_size=32, num_layers=1)
        self.head = nn.Linear(32, dict_size)

    def forward(self, x):
        hidden, h_x = self.rnn(x)
        last_hidden = hidden[-1]
        x = self.head(last_hidden)
        # x = F.softmax(input=x, dim=0)

        return x, hidden

with open("rnns/input.txt", mode='r') as file:
    word_set = set()
    for line in file:
        if line == "":
            continue
        for word in line.split():
            if word != "":
                word_set.add(word)

dict_size = len(word_set)

word_map = {}

for i, word in enumerate(word_set):
    word_map[word] = i


embedding = nn.Embedding(num_embeddings=dict_size, embedding_dim=EMBEDDING_DIM)
model = TextGenerator(dict_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    params=list(model.parameters()) + list(embedding.parameters()),
    lr=0.001
)

with open("rnns/input.txt", mode='r') as file:
    word_set = set()
    for line in file:
        word_arr = line.split()

        if len(word_arr) <= 1:
            continue

        curr_input_embedding = []

        total_loss = 0
        for i in range(len(word_arr) - 1):
            optimizer.zero_grad()

            word = word_arr[i]
            index = word_map[word]
            truth = torch.tensor(word_map[word_arr[i + 1]])

            curr_input_embedding.append(embedding(torch.tensor(index)).detach())

            new_embedding = torch.stack(curr_input_embedding).requires_grad_()

            output, hidden = model(new_embedding)

            loss = criterion(output, truth)

            total_loss += loss
            loss.backward()

            optimizer.step()

        total_loss /= (len(word_arr) - 1)

        print(total_loss.item())
