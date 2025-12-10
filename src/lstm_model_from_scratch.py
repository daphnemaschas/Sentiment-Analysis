
import torch
import torch.nn as nn
import torch.nn.functional as F

class ManualLSTMCell(nn.Module):
    """
    A manually implement LSTM cell that reproduces the behavior of nn.LSTMCell
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(input_dim + hidden_dim, 4*hidden_dim) # Each gate gets input_dim + hidden_dim inputs
    
    def forward(self, x_t, h_prev, c_prev):
        """
        Args:
            x_t: input at time t (the current encoded word)
            h_prev: hidden state from the previous timestep
            c_prev: cell state from the previous timestep
        """
        combined = torch.cat([x_t, h_prev], dim=1)  # (batch, input+hidden)
        gates = self.W(combined) # (batch, 4*hidden)

        # Split into 4 gates
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)

        i_t = torch.sigmoid(i_t) # Input gate
        f_t = torch.sigmoid(f_t) # Forget gate
        o_t = torch.sigmoid(o_t) # Output gate
        g_t = torch.tanh(g_t)

        # Update memory cell
        c_t = f_t * c_prev + i_t * g_t

        # New hidden state
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class ManualLSTM(nn.Module):
    """
    Manually implemented multi-timestep LSTM using ManualLSTMCell.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.hidden_dim = hidden_dim

        self.cell = ManualLSTMCell(embed_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: LongTensor (batch, seq_len): tensor of token indices
        """
        batch_size, seq_len = x.size()

        embedded = self.embedding(x)  # (batch, seq, embed_dim)

        # Initial zero hidden and cell states
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # Process sequence
        for t in range(seq_len):
            x_t = embedded[:, t, :]     # (batch, embed_dim)
            h_t, c_t = self.cell(x_t, h_t, c_t)

        # Classification with last hidden state
        logits = self.fc(h_t)
        return logits

