# AutoMemoryModule: A PyTorch Memory Mechanism for Large Language Models

The AutoMemoryModule is a PyTorch implementation of a dynamic memory mechanism for large language models. It retains important tokens from input sentences, improving model performance by utilizing this context. Particularly useful for tasks requiring the model to remember crucial information from previous inputs while processing new ones.

## Key Features
- **Dynamic Memory**: Adapts to input sequences, retaining only important tokens.
- **Long-term Dependency Handling**: Improves long-term context understanding in tasks.
- **Scalability**: Configurable memory size for different tasks and memory requirements.
- **Memory Efficiency**: Dynamic threshold mechanism prevents memory waste.
- **Easy Integration**: Effortlessly integrates into existing large language models as an additional component.

## Mechanism Overview
1. **Score Network**: Computes importance scores for tokens in the input sentence and memory context.
2. **Threshold Network**: Computes a dynamic threshold based on the importance scores of the current memory context.

The module processes an input sentence and a memory context, calculates importance scores for tokens, computes a dynamic threshold, retains tokens with scores above the threshold, and updates the memory context while maintaining a predefined maximum size.

## Neural Networks
- Score Network: `nn.Sequential(nn.Linear(max_memory_size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, max_memory_size), nn.Sigmoid())`
- Threshold Network: `nn.Sequential(nn.Linear(max_memory_size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())`

## Token Importance Score Calculation
1. Compute new token importance scores.
2. Compute memory context token importance scores.
3. Calculate the dynamic threshold using the threshold network.

## Memory Context Update
1. Combine tokens and importance scores.
2. Retain tokens with scores greater than or equal to the dynamic threshold.
3. Limit memory context size to the predefined maximum.

# Usage Example: Training a Simple Language Model
Train a simple large language model using the AutoMemoryModule to predict the next word in a sentence with a sample dataset.

## Training ##

```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from AutoMemoryModule import AutoMemoryModule

    # Define a sample dataset of sentences
    class SentenceDataset(Dataset):
        def __init__(self, sentences, vocab):
            self.sentences = sentences
            self.vocab = vocab

        def __len__(self):
            return len(self.sentences)

        def __getitem__(self, idx):
            sentence = self.sentences[idx]
            input_sentence = sentence[:-1]
            target_word = sentence[-1]

            input_tokens = [self.vocab[word] for word in input_sentence]
            target_token = self.vocab[target_word]

            return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_token, dtype=torch.long)

    # Define the large language model with AutoMemoryModule
    class SimpleLanguageModel(nn.Module):
        def __init__(self, vocab_size, token_dim, hidden_dim, max_memory_size):
            super(SimpleLanguageModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, token_dim)
            self.memory_module = AutoMemoryModule(token_dim, hidden_dim, max_memory_size)
            self.lstm = nn.LSTM(token_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_size)

        def forward(self, sentence_tokens, memory_context):
            embedded_tokens = self.embedding(sentence_tokens)
            memory_context, _ = self.memory_module(embedded_tokens, memory_context)
            lstm_output, _ = self.lstm(memory_context.unsqueeze(0))
            logits = self.fc(lstm_output.squeeze(0))
            return logits

    # Hyperparameters
    vocab_size = len(vocab)
    token_dim = 100
    hidden_dim = 128
    max_memory_size = 50
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 1

    # Initialize the dataset, model, optimizer, and loss function
    sentences = [...]  # List of sentences
    vocab = {...}  # Dictionary mapping words to token ids
    dataset = SentenceDataset(sentences, vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleLanguageModel(vocab_size, token_dim, hidden_dim, max_memory_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        for i, (sentence_tokens, target_token) in enumerate(dataloader):
            # Initialize the memory context
            memory_context = None

            # Forward pass
            logits = model(sentence_tokens, memory_context)

            # Calculate the loss
            loss = loss_fn(logits, target_token)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataset)}], Loss: {loss.item():.4f}')

```