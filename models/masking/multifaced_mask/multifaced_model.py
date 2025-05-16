import torch
import torch.nn as nn

class MultiFacedEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32, latent_dim=64):
        super(MultiFacedEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Flatten()
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)

        mu = self.fc_mu(encoded)

        return mu


class MultiFacedDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=32, out_channels=3):
        super(MultiFacedDecoder, self).__init__()

        # Decoder layers
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim * 4 * 10 * 10)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.final_layer = nn.Sequential(
            nn.Linear(48 * 10 * 10, out_channels * 10 * 10),
            nn.Tanh()
        )

    def forward(self, z, batch_size, time, regions):

        decoded = self.decoder_fc(z).view(batch_size * time * regions, 128, 10, 10)


        decoded = self.decoder(decoded)
        decoded = self.final_layer(decoded.view(decoded.size(0), -1))


        return decoded.view(batch_size, time, regions, 3, 10, 10)


class MultiFacedVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32, latent_dim=64, final_embed_dim=64):
        super(MultiFacedVAE, self).__init__()
        self.encoder = MultiFacedEncoder(in_channels, hidden_dim, latent_dim)
        self.decoder = MultiFacedDecoder(latent_dim, hidden_dim, in_channels)
        self.final_embed_dim = final_embed_dim

    def forward(self, x):
        batch_size, time, regions, channels, width, height = x.shape
        x = x.view(batch_size * time * regions, channels, width, height)

        mu = self.encoder(x)
        embedding = mu.view(batch_size, time, regions, self.final_embed_dim)

        reconstructed = self.decoder(mu, batch_size, time, regions)
        return reconstructed, embedding
