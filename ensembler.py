import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, hidden_size, dim_feedforward, dropout, activation="gelu"):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MultiViewAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout, **kwargs):
        super(MultiViewAttention, self).__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.qkv_projection = nn.Linear(hidden_size, hidden_size * 3)
        self.o_projection = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, video_features, video_attention_mask):
        """
        video_features: (batch_size, num_views, num_frames, hidden_size)
        video_attention_mask: (batch_size, num_views, num_frames)

        output: (batch_size, num_frames, hidden_size)
        """
        batch_size, num_views, num_frames, hidden_size = video_features.size()
        total_frames = num_views * num_frames

        video_features = video_features.view(batch_size, -1, hidden_size)
        # (batch_size, total_frames, hidden_size) (4, 320, 256)

        video_attention_mask = video_attention_mask.view(-1, total_frames)
        # (batch_size, total_frames) (4, 320)

        ### Seperate the q, k, v
        qkv = self.qkv_projection(video_features)
        # (batch_size, total_frames, hidden_size * 3)

        qkv = qkv.reshape(
            batch_size, total_frames, self.n_heads, self.head_dim * 3
        ).contiguous()

        qkv = qkv.permute(0, 2, 1, 3)
        # (batch_size, num_heads, total_frames, head_dim * 3)

        q, k, v = qkv.chunk(3, dim=-1)
        # q, k, v: (batch_size, total_frames, n_heads, head_dim)

        # calculate scaled dot product attention
        attn_out = self.scaled_dot_product_attention(q, k, v, video_attention_mask)
        # (batch_size, num_heads, total_frames, head_dim)

        attn_out = attn_out.permute(0, 2, 1, 3)
        # (batch_size, total_frames, num_heads, head_dim)

        attn_out = attn_out.reshape(batch_size, total_frames, hidden_size).contiguous()
        # (batch_size, total_frames, hidden_size)

        # output projection
        attention_output = self.o_projection(attn_out)
        # (batch_size, total_frames, hidden_size)

        # view back to original shape
        attention_output = attention_output.reshape(
            batch_size, num_views, num_frames, hidden_size
        ).contiguous()

        return attention_output

    def scaled_dot_product_attention(self, q, k, v, mask):
        """
        q: (batch_size, num_heads, seq_len, head_dim)
        k: (batch_size, num_heads, seq_len, head_dim)
        v: (batch_size, num_heads, seq_len, head_dim)
        mask: (batch_size, seq_len)

        output: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = q.size()
        mask = mask.unsqueeze(1).unsqueeze(2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (head_dim**0.5)
        scores = scores.masked_fill(mask, -1e9)

        attention = nn.functional.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        output = torch.matmul(attention, v)
        return output


class MultiViewEnsembler(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_heads,
        dropout,
        feedforward_dim,
        num_views,
        num_frames,
        **kwargs
    ):
        super(MultiViewEnsembler, self).__init__()

        self.view_position_embedding = nn.Embedding(num_views, hidden_size)
        self.frame_position_embedding = nn.Embedding(num_frames, hidden_size)

        self.attention = MultiViewAttention(hidden_size, n_heads, dropout)
        self.view_merger = nn.Linear(num_views, 1)

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.feedforward = FeedForward(hidden_size, feedforward_dim, dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.kwargs = kwargs

    def forward(self, video_features, video_attention_mask):
        """
        video_features: (batch_size, num_views, num_frames, hidden_size) (4, 5, 64, 256)
        video_attention_mask: (batch_size, num_views, num_frames)
        """
        batch_size, num_views, num_frames, hidden_size = video_features.size()
        encoded_features = self.add_position_embedding(video_features)

        attention_output = self.attention(encoded_features, video_attention_mask)
        attention_output = self.layer_norm1(attention_output + encoded_features)
        attention_output = self.dropout(attention_output)

        feedforward_output = self.feedforward(attention_output)
        feedforward_output = self.layer_norm2(feedforward_output + attention_output)
        feedforward_output = self.dropout(feedforward_output)

        feedforward_output = feedforward_output.permute(0, 2, 3, 1)
        ensembler_output = self.view_merger(feedforward_output).squeeze(-1)

        return ensembler_output

    def add_position_embedding(self, x):
        batch_size, num_views, num_frames, hidden_size = x.size()
        view_position = torch.arange(num_views).to(x.device)
        frame_position = torch.arange(num_frames).to(x.device)

        view_position = self.view_position_embedding(view_position)
        frame_position = self.frame_position_embedding(frame_position)

        view_position = view_position.unsqueeze(0).unsqueeze(2)
        frame_position = frame_position.unsqueeze(0).unsqueeze(1)

        x = x + view_position + frame_position
        return x


if __name__ == "__main__":
    batch_size = 2
    num_views = 5
    num_frames = 11
    hidden_size = 128
    num_heads = 4
    feedforward_dim = hidden_size * 2

    video_features = torch.randn(batch_size, num_views, num_frames, hidden_size)
    video_attention_mask = torch.randint(
        0, 2, (batch_size, num_views, num_frames), dtype=torch.bool
    )

    ensembler = MultiViewEnsembler(
        hidden_size, num_heads, 0.1, feedforward_dim, num_views, num_frames
    )
    ensembled_video_features = ensembler(video_features, video_attention_mask)
    print(ensembled_video_features.size())
