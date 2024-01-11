# Minimalist-GPT
Code implementation of GPT as a [finite-state markov chain](https://colab.research.google.com/drive/1SiF0KZJp75rUeetKOWqpsA8clmHP6jMg#scrollTo=eNAcd1qYxZ1P)

This repository contains a minimalist implementation of a GPT (Generative Pre-trained Transformer) model in PyTorch, named Minmalist-GPT. The implementation is kept minimal for educational purposes and includes functionalities for training on a simple binary sequence.

## Overview

Minmalist-GPT is designed to demonstrate the core concepts of the GPT architecture in a simplified manner. The implementation includes a basic GPT model, training on a binary sequence, and a visualization of the model's state transitions.

## Code Structure

The code is organized into the following main components:

- **minmalist_gpt.py**: Defines the GPT model (`GPT` class) and related configurations (`GPTConfig` class).

- **main.py**: Contains functions for training Minmalist-GPT on a binary sequence and visualizing the model's state transitions.

- **states-1.png**: Visualization of Minmalist-GPT's initial state transitions.

- **states-2.png**: Visualization of Minmalist-GPT's state transitions after training.

## Requirements

torch<br>
graphviz

## Usage

To use Minmalist-GPT, you can follow the example in `main.py`:

```python
# Example usage
from minmalist_gpt import GPT, GPTConfig, plot_model, token_seq_to_tensor, do_training

# Set the vocabulary size, context length, and other configuration parameters
vocab_size = 2
context_length = 3
config = GPTConfig(
    block_size=context_length,
    vocab_size=vocab_size,
    n_layer=4,
    n_head=4,
    n_embd=16,
    bias=False,
)

# Create a new Minmalist-GPT model
gpt = GPT(config)

# Plot the initial state transitions
dot = plot_model()
dot.render('states-1', format='png')

# Prepare training data sequence
seq = list(map(int, "111101111011110"))
print("\nTraining data sequence: ", seq)
X, Y = token_seq_to_tensor(seq)

# Train Minmalist-GPT
do_training(X, Y)

# Plot the state transitions after training
dot = plot_model()
dot.render('states-2', format='png')
```
## Acknowledgments
This implementation is inspired by the GPT architecture, and special thanks to the authors of the original GPT paper.

## License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
