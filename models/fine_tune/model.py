# This project uses Llama 3.2, a foundational large language model developed by Meta.
# Llama 3.2 is licensed under the Llama 3.2 Community License,
# Copyright © Meta Platforms, Inc. All Rights Reserved.


import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset
from torch.nn import DataParallel


class TrafficInstructionDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, raw_data):
        """
        Args:
            embeddings: Tensor of shape [dataset_size, input_length, num_regions, embed_dim].
            raw_data: Tensor of shape [dataset_size, input_length, num_regions, grid_dim, grid_dim].
        """
        self.embeddings = embeddings
        self.raw_data = raw_data

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        embedding = self.embeddings[idx].clone().detach().float()  # [input_length, num_regions, embed_dim]
        raw_data = self.raw_data[idx].clone().detach().float()  # [input_length, num_regions, grid_dim, grid_dim]
        return {"embedding": embedding, "raw_data": raw_data}


class LLMFineTuner(nn.Module):
    def __init__(self, model_name, tokenizer_name, device='cuda'):
        super(LLMFineTuner, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device


        self.embedding_to_hidden = nn.Linear(80, self.model.config.hidden_size).to(device)


        self._unfreeze_last_layers(10)

    def _unfreeze_last_layers(self, num_layers_to_unfreeze):
        total_layers = len(self.model.model.layers)
        for i in range(total_layers):
            for param in self.model.model.layers[i].parameters():
                param.requires_grad = i >= total_layers - num_layers_to_unfreeze

    def forward(self, input_embeds, attention_mask):

        outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        return outputs.logits

class RegressionHead(nn.Module):
    def __init__(self, hidden_dim=80, target_dim=(10, 10), device=None):
        super(RegressionHead, self).__init__()
        self.target_dim = target_dim

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(7, 7), stride=(5, 6), padding=(1, 1))
        self.act1 = nn.Tanh()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(6, 6), stride=(4, 5), padding=(1, 1))
        self.act2 = nn.Tanh()

        # Self-Attention Layer
        self.self_attention = SelfAttention(embed_dim=32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1))
        self.act3 = nn.Tanh()

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=(3, 3), stride=(3, 3), padding=(1, 1))
        self.act4 = nn.Tanh()

        self.fc = nn.Sequential(
            # for target length 4
            nn.Linear(357, target_dim[0] * target_dim[1]),
            # for target length 3
            # nn.Linear(476, target_dim[0] * target_dim[1]),
            # for target length 2
            # nn.Linear(714, target_dim[0] * target_dim[1]),
            # for target length 1
            # nn.Linear(1428, target_dim[0] * target_dim[1]),
            nn.Tanh()
        )


    def forward(self, x, target_length):
        batch_size, dim_channel, hidden_dim = x.shape
        # print(x.shape)

        # Add channel dimension
        x = x.unsqueeze(1)

        # Pass through Conv1
        x = self.conv1(x)
        x = self.act1(x)
        # print(x.shape)

        # Pass through Conv2
        x = self.conv2(x)
        x = self.act2(x)
        # print(x.shape)


        conv2_shape = x.shape  # [batch_size, channels, height, width]

        # Flatten and apply Self-Attention
        x = x.flatten(start_dim=2).transpose(1, 2)  # [batch_size, seq_length, embed_dim]
        x = self.self_attention(x)
        # print(x.shape)


        x = x.transpose(1, 2).reshape(conv2_shape)  # [batch_size, channels, height, width]

        # Pass through Conv3
        x = self.conv3(x)
        x = self.act3(x)

        # Pass through Conv4
        x = self.conv4(x)
        x = self.act4(x)

        # Flatten and pass through Fully Connected layers
        x = x.reshape(batch_size, target_length, -1)
        fc_output = self.fc(x)

        output = fc_output.reshape(batch_size, target_length, *self.target_dim)
        return output

    def get_shared_layers(self):
        """Provide references to shared layers."""
        return self.conv1, self.conv2, self.self_attention


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)


        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)


        out = torch.matmul(attention_weights, value)
        return out
