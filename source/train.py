import os
import json
import argparse
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import MedicareDataset
from model import MedicareRegressionModel

def _get_data_loader(batch_size, data_dir, filename):
    """Instantiate a PyTorch data loader
    """
    df = pd.read_csv(os.path.join(data_dir, filename))
    df_arr = df.to_numpy()
    
    X_train, X_val, y_train, y_val = train_test_split(df_arr[:, 1:9], df_arr[:, 0:1], test_size=0.2, stratify=df_arr[:, 9:10])
    X_train = torch.from_numpy(X_train).float()
    X_val = torch.from_numpy(X_val).float()
    y_train = torch.from_numpy(y_train).float()
    y_val = torch.from_numpy(y_val).float()
    
    ds = MedicareDataset(X_train, y_train)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dataloader

def _train(model, data_loader, epochs, optimizer, criterion, device):
    """Perform training on provided hyperparameters
    :param model: PyTorch model to train
    :param train_loader: PyTorch DataLoader that should be used during training.
    :param epochs: Total number of epochs to train for.
    :param optimizer: Optimizer to use during training.
    :param criterion: Loss function to optimize. 
    :param device: Where the model and data should be loaded (gpu or cpu).
    """
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad() # Zero accumulated gradients
            output = model(data) # Forward propagation
            loss = criterion(output, target) # Calculate loss
            loss.backward() # Backward propagation
            optimizer.step()
            total_loss += loss.item()

        # print loss stats
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(data_loader)))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    # Custom parameters that we can pass from notebook instance
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--input_dim', type=int, default=2, metavar='ID', help='input dimension for training data')
    parser.add_argument('--output_dim', type=int, default=1, metavar='OD', help='output dimension for neural network')   

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_loader = _get_data_loader(args.batch_size, args.data_dir, 'medicare.csv') 

    model = MedicareRegressionModel(args.input_dim, args.output_dim)
    # model = model.to(torch.double)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    _train(model, train_data_loader, args.epochs, optimizer, criterion, device)
    
    path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)
    
