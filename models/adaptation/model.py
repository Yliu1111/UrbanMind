import torch
import torch.nn as nn

class RecoveryHead(nn.Module):
    def __init__(self, shared_conv1, shared_conv2, share_attention, llm_embed_dim):
        super(RecoveryHead, self).__init__()
        self.shared_conv1 = shared_conv1
        self.shared_conv2 = shared_conv2
        self.self_attention = share_attention

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64,
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.act3 = nn.SiLU()

        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=45,
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.act4 = nn.SiLU()

        self.attention_layer = nn.MultiheadAttention(
            embed_dim=45,
            num_heads=9
        )

        self.output_fc = nn.Linear(1069, llm_embed_dim)
        self.output_act = nn.SiLU()

    def forward(self, x):
        batch_size, seq_length, llm_embed_dim = x.size()

        x = x.unsqueeze(1)

        x = torch.tanh(self.shared_conv1(x))
        x = torch.tanh(self.shared_conv2(x))

        conv2_shape = x.shape

        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.self_attention(x)

        x = x.transpose(1, 2).reshape(conv2_shape)

        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(2)
        x = x.permute(2, 0, 1)

        x, _ = self.attention_layer(x, x, x)

        x = x.permute(1, 0, 2)
        x = x.reshape(batch_size, seq_length, -1)

        x = self.output_fc(x)
        x = self.output_act(x)

        return x
