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
        # Residual: include neighbour raw state (5) + embedding (E)
        self.fc = nn.Sequential(
            nn.Linear(8 + 2 * embed_dims, final_fc_dims),
            nn.LayerNorm(final_fc_dims),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(final_fc_dims, 1),
        )

    def forward(self, x):
        """
        x: (B, state_dim+1) with appended action index
        Returns: (B,1) Q-value for selected action
        """
        cur_pos = x[:, :3]
        actions = x[:, -1].long()
        B = x.size(0)
        neighbours = x[:, 3:-1].reshape(B, -1, 5)
        N = neighbours.size(1)

        # DeepSet embeddings
        cur_pos_rep = cur_pos.unsqueeze(1).expand(-1, N, -1)  # (B,N,3)
        ds_input = torch.cat((cur_pos_rep, neighbours), dim=2)  # (B,N,8)
        embeds = self.deep_set(ds_input.view(-1, ds_input.size(-1)))  # (B*N,E)
        embeds = embeds.view(B, N, -1)  # (B,N,E)

        # Mask invalid neighbours
        mask = torch.any(neighbours != 0, dim=2)  # (B,N)
        masked_embeds = embeds * mask.unsqueeze(2).float()
        valid_counts = mask.sum(dim=1, keepdim=True).float()
        pooled_embeds = masked_embeds.sum(dim=1) / torch.clamp(valid_counts, min=1.0)

        # Select chosen neighbour’s embed + raw state
        chosen_embeds = embeds[torch.arange(B), actions]  # (B,E)
        chosen_states = neighbours[torch.arange(B), actions]  # (B,5)

        # Fusion: [cur_pos | chosen_embeds | pooled_embeds | chosen_states]
        fc_input = torch.cat(
            (cur_pos, chosen_embeds, pooled_embeds, chosen_states), dim=1
        )
        return self.fc(fc_input)

    def predict(self, states):
        """
        Predict Q-values Q(s,a) for all actions a (batched).
        Invalid actions padded with -inf.
        Args:
            states: (B,S) or (S,)
        Returns:
            q_all: (B,n_actions) or (n_actions,)
        """
        if states.dim() == 1:
            states = states.unsqueeze(0)
        B = states.size(0)
        N = system_config["satellite"]["n_neighbours"]

        # Expand for all possible actions
        actions = torch.arange(N, device=states.device).repeat(B, 1)  # (B,N)
        states_rep = states.unsqueeze(1).repeat(1, N, 1)  # (B,N,S)
        sa_pairs = torch.cat(
            (states_rep, actions.unsqueeze(2).float()), dim=2
        )  # (B,N,S+1)

        # Flatten to batch
        sa_pairs = sa_pairs.view(B * N, -1)
        with torch.no_grad():
            self.eval()
            q_vals = self.forward(sa_pairs).view(B, N)  # (B,N)

        # Mask invalid neighbours
        neighbours = states[:, 3:].reshape(B, N, 5)
        mask = torch.any(neighbours != 0, dim=2)  # (B,N)
        q_vals[~mask] = -float("inf")

        return q_vals.squeeze(0) if B == 1 else q_vals

    def fit(self, states, actions, targets, lr, epochs=1):
        """
        Train the network on (s,a) → target Q(s,a).
        Args:
            states: (B,S)
            actions: (B,) Long indices
            targets: (B,1) TD targets
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.SmoothL1Loss()
        avg_loss = 0.0
        self.train()

        for _ in range(epochs):
            optimizer.zero_grad()
            # Build (s,a) input
            sa_pairs = torch.cat(
                (states, actions.unsqueeze(1).float()), dim=1
            )  # (B,S+1)
            q_pred = self.forward(sa_pairs)  # (B,1)
            loss = loss_fn(q_pred, targets)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        return avg_loss / epochs


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
    def __init__(self, input_dims, mem_size=2048, target_update_interval=10):
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

        self.q_target = QNetwork(
            ds_input_dims=8,
            ds_fc1_dims=agent_config["dqn"]["deepset"]["fc1_dims"],
            ds_fc2_dims=agent_config["dqn"]["deepset"]["fc2_dims"],
            embed_dims=agent_config["dqn"]["deepset"]["embedding_dims"],
            final_fc_dims=agent_config["dqn"]["final_fc_dims"],
            dropout=agent_config["dqn"]["dropout"],
        ).to(DEVICE)

        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.learn_step_counter = 0
        self.target_update_interval = target_update_interval

    def update_target_network(self):
        self.q_target.load_state_dict(self.q_eval.state_dict())

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)

    def choose_action(self, state):
        rand = np.random.random()
        if not isinstance(state, torch.Tensor):
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        else:
            state_t = state.to(DEVICE, dtype=torch.float32)

        if rand < self.epsilon:
            neighbour_states = state_t[3:].cpu().numpy().reshape(-1, 5)
            valid_actions = np.where(np.any(neighbour_states != 0, axis=1))[0]
            action = int(np.random.choice(valid_actions))
        else:
            q_values = self.q_eval.predict(state_t)
            action = int(torch.argmax(q_values).item())
        return action

    def learn(self):
        if self.memory.mem_counter >= self.batch_size:
            # 1. Sample batch
            states, actions_idx, rewards, new_states, not_done = (
                self.memory.sample_buffer(self.batch_size)
            )

            # 2. To device
            states = states.to(DEVICE).float()
            new_states = new_states.to(DEVICE).float()
            rewards = rewards.to(DEVICE).float().unsqueeze(1)
            not_done = not_done.to(DEVICE).float().unsqueeze(1)
            actions_idx = actions_idx.to(DEVICE).long()  # (B,)

            # 3. Target network for Q(s',a')
            with torch.no_grad():
                q_next = (
                    self.q_target.predict(new_states).max(dim=1, keepdim=True).values
                )

            q_target = rewards + self.gamma * q_next * not_done

            # 4. Train eval net
            _ = self.q_eval.fit(states, actions_idx, q_target, lr=self.lr)

            # 5. Target network sync
            self.learn_step_counter += 1
            if self.learn_step_counter % self.target_update_interval == 0:
                self.update_target_network()
