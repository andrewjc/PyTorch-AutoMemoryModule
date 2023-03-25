import torch
import torch.nn as nn

"""
Title: AutoMemoryModule.py
Author: Andrew Cranston <andrew.cranston@gmail.com>

Description:
    AutoMemoryModule is a PyTorch-based memory mechanism designed to improve
    large language model performance by selectively retaining important tokens.
    This module uses two neural networks, score_net and threshold_net, to assign
    importance scores to tokens and determine a dynamic threshold for token retention.
    The primary goal is to maintain an up-to-date memory context, making it ideal for
    various NLP tasks such as sentiment analysis, text classification, and translation.
"""

class AutoMemoryModule(nn.Module):
    def __init__(self, token_dim, hidden_dim, max_memory_size, padding_token=0):
        super(AutoMemoryModule, self).__init__()
        self.padding_token = padding_token
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.max_memory_size = max_memory_size

        self.importance = torch.randn(0)

        self.score_net = nn.Sequential(
            nn.Linear(max_memory_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_memory_size),
            nn.Sigmoid(),
        )

        self.threshold_net = nn.Sequential(
            nn.Linear(max_memory_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, sentence_tokens, memory_context):
        sentence_tokens = sentence_tokens.float()
        padded_sentence_tokens = torch.cat([sentence_tokens, torch.zeros(self.max_memory_size - sentence_tokens.shape[0])])

        # Compute masks for sentence_tokens to ignore padding tokens during score computation
        sentence_tokens_mask = (padded_sentence_tokens != self.padding_token)

        # Compute importance scores for the new tokens
        # pad sentence_tokens to the max_memory_size
        new_importance_scores = self.score_net(padded_sentence_tokens).squeeze()
        new_importance_scores = (new_importance_scores * sentence_tokens_mask.float()).squeeze()

        # Check if memory_context is not None, then compute importance scores and mask
        if memory_context is not None:
            memory_context = memory_context.float()

            memory_context_mask = (memory_context != self.padding_token).unsqueeze(-1)
            current_importance_scores = self.score_net(memory_context).squeeze()
            current_importance_scores = (current_importance_scores * memory_context_mask.float()).squeeze()
        else:
            current_importance_scores = torch.zeros_like(new_importance_scores)
            memory_context = torch.zeros(self.max_memory_size)

        # Calculate the dynamic threshold for retaining tokens

        threshold_factor = self.threshold_net(current_importance_scores).item()

        # Combine new tokens and current memory context, along with their importance scores
        combined_tokens = torch.cat([memory_context, padded_sentence_tokens], dim=0)
        combined_importance = torch.cat([current_importance_scores, new_importance_scores], dim=0)

        # Filter the tokens based on the dynamic threshold
        mask = combined_importance >= threshold_factor

        # Update the memory context and importance
        memory_context = combined_tokens[mask]
        self.importance = combined_importance[mask]

        # Limit the size of the memory context if it exceeds the maximum size
        if memory_context.size(0) > self.max_memory_size:
            sorted_indices = torch.argsort(self.importance, descending=True)
            memory_context = memory_context[sorted_indices[:self.max_memory_size]]
            self.importance = self.importance[sorted_indices[:self.max_memory_size]]
        else:
            memory_context = torch.cat([memory_context, torch.zeros(self.max_memory_size - memory_context.shape[0])])
            combined_importance = torch.cat([self.importance, torch.zeros(self.max_memory_size - self.importance.shape[0])])

        return memory_context.clone().detach(), combined_importance.clone().detach()
