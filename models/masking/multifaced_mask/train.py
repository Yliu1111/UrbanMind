import sys
sys.path.append('../')
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from util import *
from multifaced_model import MultiFacedVAE

def vae_masked_loss_function(recon_data, original_data, mask):
    recon_loss = nn.MSELoss(reduction='none')(recon_data, original_data)
    masked_recon_loss = (recon_loss * mask).sum() / mask.sum()
    return masked_recon_loss

def multifaced_train(data_paths, epochs=1000, lr=0.001, batch_size=32, mask_ratios=(1, 1, 2), device=None):

    speed = torch.tensor(loaddata(data_paths['speed'])).to(device)
    inflow = torch.tensor(loaddata(data_paths['inflow'])).to(device)
    demand = torch.tensor(loaddata(data_paths['demand'])).to(device)
    speed, demand, inflow = process_d(speed, demand, inflow)

    speed = speed.reshape(-1, 12, 63, 10, 10)[:, :, :50, :, :]
    inflow = inflow.reshape(-1, 12, 63, 10, 10)[:, :, :50, :, :]
    demand = demand.reshape(-1, 12, 63, 10, 10)[:, :, :50, :, :]
    batch_data = torch.stack([speed, inflow, demand], dim=3)


    dataset = MyDataset(batch_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae = MultiFacedVAE(in_channels=3, hidden_dim=32, latent_dim=64).to(device)
    if torch.cuda.device_count() > 1:
        vae = nn.DataParallel(vae)

    optimizer = optim.Adam(vae.parameters(), lr=lr)

    temporal_epochs = 25
    spatial_epochs = 25
    agnostic_epochs = epochs - temporal_epochs - spatial_epochs
    save_dir = "./model"
    os.makedirs(save_dir, exist_ok=True)


    spatial_points = 90
    temporal_hours = 2

    for epoch in range(epochs):
        total_loss = 0
        vae.train()

        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            epoch_within_100 = epoch % 100
            if epoch_within_100 < temporal_epochs:
                masked_data, mask = apply_mask(batch_data, mask_type="temporal", num_channels=3, temporal_hours=temporal_hours, return_mask=True)
                if epoch_within_100 == 0:
                    print("Entering Temporal Masking Strategy")

            elif epoch_within_100 < temporal_epochs + spatial_epochs:
                masked_data, mask = apply_mask(batch_data, mask_type="spatial", num_channels=3, spatial_points=spatial_points, return_mask=True)
                if epoch_within_100 == 25:
                    print("Entering Spatial Masking Strategy")

            else:
                masked_data, mask = apply_mask(batch_data, mask_type="agnostic", num_channels=3,
                                               temporal_hours=temporal_hours, spatial_points=spatial_points, return_mask=True)
                if epoch_within_100 == 50:
                    print("Entering Agnostic Masking Strategy")


            recon_data, embedding = vae(masked_data)
            loss = vae_masked_loss_function(recon_data, batch_data, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")


        if (epoch + 1) % 200 == 0:
            torch.save(vae.state_dict(), os.path.join(save_dir, f'multifaced_mask_t{temporal_hours}_s{spatial_points}_epoch_{epoch + 1}.pth'))

    if epoch == epochs - 1:
        embedding_mean = embedding.mean().item()
        embedding_max = embedding.max().item()
        embedding_min = embedding.min().item()
        print(f"Final Epoch Embedding - Mean: {embedding_mean:.4f}, Max: {embedding_max:.4f}, Min: {embedding_min:.4f}")

    return vae




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_paths = {
        'speed': '../../../speed_SZ.csv',
        'inflow': '../../../inflow_SZ.csv',
        'demand': '../../../demand_SZ.csv'
    }

    epochs = 1000
    lr = 0.001
    batch_size = 32
    mask_ratios = (1, 1, 2)

    trained_model = multifaced_train(data_paths, epochs, lr, batch_size, mask_ratios, device)


if __name__ == '__main__':
    main()
