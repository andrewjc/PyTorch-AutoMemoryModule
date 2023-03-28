import torch
import torch.nn as nn


# LLM auto memory module
# Provide a mechanism for self pruning of the memory context.
#
# Score Net:
# - Responsible for computing the importance scores for both the new sentence
#   tokens and the current memory context.
#
# Threshold Net:
# - Calculates a dynamic threshold factor based on the mean of the memory context.
#   The threshold is then used to filter the combined tokens, retaining only the ones
#   above the threshold.

# The memory context is updated with the filtered tokens and their corresponding
# importance scores.
#
# If the memory context exceeds the maximum size, it is trimmed down to the top most
# important tokens.

class AutoMemoryModule(nn.Module):
    def __init__(self, embedding_layer, max_sentence_length, max_memory_context, embedding_size, padding_token, device='cpu'):
        super().__init__()
        self.vocab_size = 32000
        self.max_sentence_length = max_sentence_length
        self.max_memory_context = max_memory_context
        self.embedding_size = embedding_size
        self.device = device
        self.padding_token = padding_token
        self.embedding = embedding_layer

        self.score_net = nn.Sequential(
            nn.Linear(self.embedding_size * (self.max_sentence_length), 64),
            nn.ReLU(),
            nn.Linear(64, max_sentence_length),
            nn.Sigmoid(),
        ).to(device=device)

    def forward(self, input_tokens, memory_context):
        if memory_context is None:
            memory_context = torch.zeros(self.max_memory_context, dtype=torch.long).to(device=self.device)
            # fill memory context with padding tokens
            memory_context.fill_(self.padding_token)

        input_tokens = input_tokens.to(device=self.device)
        padded_input_tokens = nn.functional.pad(input_tokens, (0, self.max_sentence_length - input_tokens.shape[-1]),
                                                value=self.padding_token)

        # score the padding_input_tokens
        input_tokens_embedding = self.embedding(padded_input_tokens).to(device=self.device).view(1, -1)
        input_tokens_scoring = self.score_net(input_tokens_embedding).squeeze(dim=0)

        memory_context_embedding = self.embedding(memory_context).to(device=self.device).view(1, -1)
        memory_context_scoring = self.score_net(memory_context_embedding).squeeze(dim=0)

        # filter out the padding tokens from the padded input tokens and their scores

        padding_token_idx = torch.nonzero(padded_input_tokens != self.padding_token).squeeze(dim=1)
        filtered_input_tokens = padded_input_tokens[padding_token_idx]
        filtered_input_tokens_scoring = input_tokens_scoring[padding_token_idx]

        ctx_padding_token_idx = torch.nonzero(memory_context != self.padding_token).squeeze(dim=1)
        filtered_memory_context = memory_context[ctx_padding_token_idx]
        filtered_memory_context_scoring = torch.index_select(memory_context_scoring, 0, ctx_padding_token_idx)

        # combine the filtered input tokens and their scores with the memory context
        # and their scores
        combined_tokens = torch.cat((filtered_input_tokens, filtered_memory_context), dim=0)
        scores = torch.cat((filtered_input_tokens_scoring, filtered_memory_context_scoring), dim=0)

        # remove duplicate tokens and their scores
        unique_tokens, indices = torch.unique(combined_tokens, return_inverse=True)
        unique_scores = torch.full_like(unique_tokens, -1e20, dtype=scores.dtype)
        unique_scores = unique_scores.scatter(0, indices, scores)

        # sort the combined tokens and their scores by the scores
        sorted_scores, sorted_indices = torch.sort(unique_scores, descending=True)
        sorted_combined_tokens = unique_tokens[sorted_indices]

        # trim the combined tokens and their scores to the max memory context size
        trimmed_combined_tokens = sorted_combined_tokens[:self.max_memory_context]
        trimmed_scores = sorted_scores[:self.max_memory_context]

        # pad the trimmed tokens and their scores with padding tokens and -1e20 respectively
        trimmed_combined_tokens = nn.functional.pad(trimmed_combined_tokens, (0, self.max_memory_context - trimmed_combined_tokens.shape[-1]), value=self.padding_token)
        trimmed_scores = nn.functional.pad(trimmed_scores, (0, self.max_memory_context - trimmed_scores.shape[-1]), value=-1e20)

        return trimmed_combined_tokens, trimmed_scores
