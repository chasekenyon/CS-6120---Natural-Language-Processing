from torch import nn
from data import *
from mlp_model import MultilayerPerceptron
import random
import time
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt

n_hidden = 128
n_epochs = 100000
print_every = 5000
plot_every = 1000
learning_rate = 0.005

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1)
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train_model(hidden_dims, learning_rate, n_epochs, checkpoints):
    mlp = MultilayerPerceptron(input_dim=n_letters, hidden_dims=hidden_dims, output_dim=n_categories)
    optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    current_loss = 0
    all_losses = []

    start = time.time()

    for epoch in range(1, n_epochs + 1):
        category, line, category_tensor, line_tensor = randomTrainingPair()
        tensor_1d = line_tensor.sum(0)

        optimizer.zero_grad()
        output = mlp(tensor_1d)
        loss = criterion(output, category_tensor)
        loss.backward()
        optimizer.step()

        current_loss += loss.item()

        if epoch % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

        if epoch in checkpoints:
            torch.save(mlp, f'mlp-classification-{len(hidden_dims)}layers-{hidden_dims[0]}nodes-{epoch}epochs.pt')

    torch.save(mlp, f'mlp-classification-{len(hidden_dims)}layers-{hidden_dims[0]}nodes.pt')

    print(all_losses)
    plt.figure()
    plt.plot(all_losses)


hidden_layer_sizes = [20, 50, 100]
hidden_layers = [1, 2, 3]
checkpoints = [10000, 50000, 100000]

for size in hidden_layer_sizes:
    for layers in hidden_layers:
        hidden_dims = [size] * layers
        train_model(hidden_dims=hidden_dims, learning_rate=learning_rate, n_epochs=n_epochs, checkpoints=checkpoints)
