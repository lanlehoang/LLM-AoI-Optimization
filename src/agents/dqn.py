import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepSetNetwork(nn.Module):
    """
    Implement the \phi function in DeepSet for processing satellite and neighbour information.
    Input to the DeepSet:
    - Current satellite relative position to destination (3,) (as contextual input)
    - Each neighbour is represented as a vector of shape (5,)
    Output:
    - Embedding vector of shape (output_dim,)
    """

    def __init__(self, input_dims, fc1_dims, fc2_dims, output_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc2_dims, output_dim)

    def forward(self, x):
        x = f.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = f.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class QNetwork(nn.Module):
    """
    Adopt DeepSet architecture to ensure permutation invariance among neighbours states and actions.
    Input shape:
    - Relative position of current satellite to destination satellite (3,)
    - For each neighbour:
        - Relative position of neighbour to destination satellite (3,)
        - Processing rate (1,)
        - Queue length (1,)
    - Action: (1,)
    """

    def __init__(
        self,
        ds_input_dims,
        ds_fc1_dims,
        ds_fc2_dims,
        embed_dims,
        final_fc_dims,
        dropout,
    ):
        super().__init__()
        self.deep_set = DeepSetNetwork(
            input_dims=ds_input_dims,
            fc1_dims=ds_fc1_dims,
            fc2_dims=ds_fc2_dims,
            output_dim=embed_dims,
            dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * embed_dims + 3, final_fc_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_fc_dims, 1),
        )

    def forward(self, x):
        """
        X shape: (batch_size, input_dims)
        """
        # Define input for DeepSet
        cur_pos = x[:, :3]
        actions = x[:, -1]
        batch_size = x.shape[0]
        neighbours = x[:, 3:-1].reshape(batch_size, -1, 5)
        n_neighbours = neighbours.shape[1]
        # Concat cur_pos to every single neighbour
        cur_pos_repeated = cur_pos.unsqueeze(1).repeat(1, n_neighbours, 1)
        x = torch.cat((cur_pos_repeated, neighbours), dim=2)

        # Run through DeepSet network
        # Embeddings of shape (batch size, n_neighbours, embed_dim)
        embeds = self.deep_set(x)
        pooled_embeds = torch.mean(embeds, dim=1)  # \rho function in DeepSet

        # Concat cur_pos, embeddings from DeepSet, and average embeddings
        selected_embeds = embeds[torch.arange(batch_size), actions.long(), :]
        x = torch.cat((cur_pos, selected_embeds, pooled_embeds), dim=1)

        # Final dense layers
        x = self.fc(x)
        return x
