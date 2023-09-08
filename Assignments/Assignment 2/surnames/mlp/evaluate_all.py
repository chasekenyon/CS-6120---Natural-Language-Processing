import torch
from mlp_model import MultilayerPerceptron
from data import *
import matplotlib.pyplot as plt
import random

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) 
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def evaluate_model(model_path, n_samples=1000):
    mlp = torch.load(model_path)
    n_correct = 0
    n_total = 0

    for i in range(n_samples):
        category, line, _, line_tensor = randomTrainingExample()
        output = mlp(line_tensor.sum(0))
        guess, _ = categoryFromOutput(output)

        if guess == category:
            n_correct += 1

        n_total += 1

    accuracy = n_correct / n_total
    return accuracy

# Model Parameters
hidden_layer_sizes = [20, 50, 100]
hidden_layers = [1, 2, 3]
checkpoints = [10000, 50000, 100000]

model_results = {}

# loop through model parameters and save results
for size in hidden_layer_sizes:
    for layers in hidden_layers:
        for checkpoint in checkpoints:
            model_name = f"{layers}-L, {size}N, {checkpoint//1000}k E"
            model_path = f"mlp-classification-{layers}layers-{size}nodes-{checkpoint}epochs.pt"
            accuracy = evaluate_model(model_path)
            model_results[model_name] = accuracy

model_names = list(model_results.keys())
accuracies = [model_results[name] for name in model_names]

plt.figure(figsize=(12, 6))
plt.bar(model_names, accuracies)
plt.xlabel('Model configuration')
plt.ylabel('Accuracy')
plt.xticks(rotation=90)
plt.title('Model accuracies')
plt.tight_layout()
plt.show()
