import torch
import torch.nn as nn
import sys

sys.path.append('../multifaced_mask')
from multifaced_model import MultiFacedEncoder, MultiFacedDecoder, MultiFacedVAE


class SingleMaskEncoder(nn.Module):
    def __init__(self, pre_trained_encoder, embed_dim=64, final_embed_dim=16):
        super(SingleMaskEncoder, self).__init__()
        self.shared_encoder = pre_trained_encoder
        for param in self.shared_encoder.parameters():
            param.requires_grad = True
        self.final_embed_dim = final_embed_dim


        self.speed_specific_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Tanh(),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Tanh(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Tanh(),
            nn.Linear(64, final_embed_dim),
            nn.Tanh()
        )
        self.inflow_specific_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Tanh(),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Tanh(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Tanh(),
            nn.Linear(64, final_embed_dim),
            nn.Tanh()
        )
        self.demand_specific_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Tanh(),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Tanh(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Tanh(),
            nn.Linear(64, final_embed_dim),
            nn.Tanh()
        )

    def forward(self, x):

        shared_embedding = self.shared_encoder(x)

        shared_embedding = shared_embedding.view(-1, 1, 12, 64)


        speed_embedding = self.speed_specific_layers(shared_embedding)
        inflow_embedding = self.inflow_specific_layers(shared_embedding)
        demand_embedding = self.demand_specific_layers(shared_embedding)
        # print(speed_embedding.shape)
        return speed_embedding, inflow_embedding, demand_embedding


class SingleMaskDecoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=32, target_dim=100):
        super(SingleMaskDecoder, self).__init__()


        self.speed_decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(1 * 4 * 4, 1 * 10 * 10),
            nn.Tanh()
        )

        self.inflow_decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(1 * 4 * 4, 1 * 10 * 10),
            nn.Tanh()
        )

        self.demand_decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(1 * 4 * 4, 1 * 10 * 10),
            nn.Tanh()
        )

    def forward(self, z, batch_size, time, regions):

        z = z.view(batch_size * time * regions, -1, 1, 1)


        speed_reconstructed = self.speed_decoder(z)
        inflow_reconstructed = self.inflow_decoder(z)
        demand_reconstructed = self.demand_decoder(z)

        return speed_reconstructed, inflow_reconstructed, demand_reconstructed


class SingleMaskVAE(nn.Module):
    def __init__(self, pre_trained_vae, final_embed_dim=16):
        super(SingleMaskVAE, self).__init__()
        self.encoder = SingleMaskEncoder(pre_trained_vae.encoder, embed_dim=64, final_embed_dim=final_embed_dim)
        self.decoder = SingleMaskDecoder(latent_dim=final_embed_dim)

    def forward(self, x):
        batch_size, time, regions, channels, width, height = x.shape
        x = x.view(batch_size * time * regions, channels, width, height)


        speed_embedding, inflow_embedding, demand_embedding = self.encoder(x)

        speed_embedding_reshaped = speed_embedding.view(batch_size * time * regions, -1, 1, 1)
        inflow_embedding_reshaped = inflow_embedding.view(batch_size * time * regions, -1, 1, 1)
        demand_embedding_reshaped = demand_embedding.view(batch_size * time * regions, -1, 1, 1)


        speed_embedding = speed_embedding.view(batch_size, time, regions, 1, 16)
        inflow_embedding = inflow_embedding.view(batch_size, time, regions, 1, 16)
        demand_embedding = demand_embedding.view(batch_size, time, regions, 1, 16)

        speed_reconstructed_flat = self.decoder.speed_decoder(speed_embedding_reshaped)
        inflow_reconstructed_flat = self.decoder.inflow_decoder(inflow_embedding_reshaped)
        demand_reconstructed_flat = self.decoder.demand_decoder(demand_embedding_reshaped)


        current_batch_size = speed_reconstructed_flat.shape[0] // (time * regions)
        speed_reconstructed = speed_reconstructed_flat.view(current_batch_size, time, regions, 1, 10, 10)
        inflow_reconstructed = inflow_reconstructed_flat.view(current_batch_size, time, regions, 1, 10, 10)
        demand_reconstructed = demand_reconstructed_flat.view(current_batch_size, time, regions, 1, 10, 10)


        speed_embedding = speed_embedding.view(current_batch_size, time, regions, -1)

        inflow_embedding = inflow_embedding.view(current_batch_size, time, regions, -1)
        demand_embedding = demand_embedding.view(current_batch_size, time, regions, -1)

        return (
            (speed_reconstructed, speed_embedding),
            (inflow_reconstructed, inflow_embedding),
            (demand_reconstructed, demand_embedding)
        )
