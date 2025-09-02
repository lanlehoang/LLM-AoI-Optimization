import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from src.utils.get_config import get_agent_config, get_system_config

agent_config = get_agent_config()
system_config = get_system_config()

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
        x = f.leaky_relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = f.leaky_relu(self.ln2(self.fc2(x)))
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
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_fc_dims, 1),
        )

    def forward(self, x):
        """
        X shape: (batch_size, input_dims)
        Returns single Q-value for the selected action
        """
        # Define input for DeepSet
        cur_pos = x[:, :3]
        actions = x[:, -1]
        batch_size = x.shape[0]
        neighbours = x[:, 3:-1].reshape(batch_size, -1, 5)
        n_neighbours = neighbours.shape[1]

        # Concat cur_pos to every single neighbour
        cur_pos_repeated = cur_pos.unsqueeze(1).repeat(1, n_neighbours, 1)
        deepset_input = torch.cat((cur_pos_repeated, neighbours), dim=2)

        # Run through DeepSet network
        # Embeddings of shape (batch size, n_neighbours, embed_dim)
        embeds = self.deep_set(deepset_input)

        # Create mask for valid neighbors (non-zero neighbor states)
        mask = torch.any(neighbours != 0, dim=2)  # Shape: (batch_size, n_neighbours)

        # Apply mask to embeddings and compute masked pooling
        masked_embeds = embeds * mask.unsqueeze(2).float()
        valid_counts = mask.sum(dim=1, keepdim=True).float()
        pooled_embeds = masked_embeds.sum(dim=1) / torch.clamp(valid_counts, min=1)

        # Concat cur_pos, selected embedding, and pooled embeddings
        selected_embeds = embeds[torch.arange(batch_size), actions.long(), :]
        fc_input = torch.cat((cur_pos, selected_embeds, pooled_embeds), dim=1)

        # Final dense layers
        x = self.fc(fc_input)
        return x

    def predict(self, states):
        """
        Predict Q-values Q(s, a) for all actions a (batched).
        Invalid actions are padded with -inf.

        Args:
            states: (B, state_dim) or (state_dim,)
        Returns:
            q_all: (B, n_actions) if batched, else (n_actions,)
        """
        if states.dim() == 1:
            states = states.unsqueeze(0)  # (1, S)
        B = states.size(0)
        n_actions = system_config["satellite"]["n_neighbours"]

        q_all = torch.full((B, n_actions), -float("inf"), device=states.device)

        with torch.no_grad():
            self.eval()
            for b in range(B):
                state = states[b]
                neighbour_states = state[3:].reshape(-1, 5)
                valid_actions = torch.where(torch.any(neighbour_states != 0, dim=1))[0]

                if valid_actions.numel() > 0:
                    x = state.repeat(valid_actions.shape[0], 1)
                    x = torch.cat((x, valid_actions.unsqueeze(1).float()), dim=1)
                    q_valid = self.forward(x).squeeze(-1)  # (n_valid,)
                    q_all[b, valid_actions] = q_valid

        return q_all.squeeze(0) if B == 1 else q_all

    def fit(self, X, y, lr, epochs=1):
        """
        X: (batch_size, input_dims): Concatenation of states and actions
        y: (batch_size, 1))
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.SmoothL1Loss()
        avg_loss = 0
        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            y_pred = self.forward(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        avg_loss /= epochs
        return avg_loss


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_counter = 0
        # Input shape of the environment
        self.input_shape = input_shape
        self.state_memory = torch.zeros((self.mem_size, input_shape))
        self.new_state_memory = torch.zeros((self.mem_size, input_shape))
        self.action_memory = torch.zeros((self.mem_size,), dtype=torch.int64)
        self.reward_memory = torch.zeros(self.mem_size)
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.float32)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = torch.as_tensor(state, dtype=torch.float32)
        self.new_state_memory[index] = torch.as_tensor(new_state, dtype=torch.float32)
        self.reward_memory[index] = float(reward)
        self.terminal_memory[index] = 1.0 - float(int(done))
        # Store integer action directly
        self.action_memory[index] = int(action)
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal


class Agent:
    def __init__(self, input_dims, mem_size=1500):
        n_actions = system_config["satellite"]["n_neighbours"]
        self.action_space = np.arange(n_actions)
        self.gamma = agent_config["train"]["gamma"]
        self.epsilon = agent_config["train"]["epsilon"]["init"]
        self.epsilon_dec = agent_config["train"]["epsilon"]["decay"]
        self.epsilon_min = agent_config["train"]["epsilon"]["min"]
        self.batch_size = agent_config["train"]["batch_size"]
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.lr = agent_config["train"]["lr"]
        self.q_eval = QNetwork(
            ds_input_dims=8,
            ds_fc1_dims=agent_config["dqn"]["deepset"]["fc1_dims"],
            ds_fc2_dims=agent_config["dqn"]["deepset"]["fc2_dims"],
            embed_dims=agent_config["dqn"]["deepset"]["embedding_dims"],
            final_fc_dims=agent_config["dqn"]["final_fc_dims"],
            dropout=agent_config["dqn"]["dropout"],
        ).to(DEVICE)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)

    # Epsilon-greedy policy
    def choose_action(self, state):
        rand = np.random.random()
        # ensure state is a torch tensor on DEVICE for predict
        if not isinstance(state, torch.Tensor):
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        else:
            state_t = state.to(DEVICE, dtype=torch.float32)

        if rand < self.epsilon:
            neighbour_states = state_t[3:].cpu().numpy().reshape(-1, 5)
            valid_actions = np.where(np.any(neighbour_states != 0, axis=1))[0]
            action = int(np.random.choice(valid_actions))
        else:
            q_values = self.q_eval.predict(state_t)  # returns torch tensor (n_actions,)
            action = int(torch.argmax(q_values).item())
        return action

    def learn(self):
        # Start training when there are sufficient samples
        if self.memory.mem_counter >= self.batch_size:
            # 1) Sample
            states, actions_idx, rewards, new_states, not_done = (
                self.memory.sample_buffer(self.batch_size)
            )

            # 2) To device & shapes
            states = states.to(DEVICE).float()  # (B, S)
            new_states = new_states.to(DEVICE).float()  # (B, S)
            rewards = rewards.to(DEVICE).float().unsqueeze(1)  # (B, 1)
            not_done = not_done.to(DEVICE).float().unsqueeze(1)  # (B, 1)
            actions_idx = actions_idx.to(DEVICE).long().unsqueeze(1)  # (B, 1)

            # 3) Build (s,a) pairs and get q_eval (used by fit)
            sa_pairs = torch.cat((states, actions_idx.float()), dim=1)  # (B, S+1)

            # 4) max_a' Q(s',a') using batched predict (returns (B, n_actions))
            q_next = (
                self.q_eval.predict(new_states).max(dim=1, keepdim=True).values
            )  # (B, 1)

            # 5) TD target (your terminal memory stores 1 - done)
            q_target = rewards + self.gamma * q_next * not_done  # (B, 1)

            # 6) single optimization step via existing fit()
            _ = self.q_eval.fit(sa_pairs, q_target, lr=self.lr)

    def save_model(self):
        pass

    def load_model(self):
        pass
