
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='GNN Recommender System')
    
    parser.add_argument('--wandbNameSuffix', type=str, default="defaultWandBName", help='Name on wandb added to run')
    
    
    
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--hidden_channels', type=int, default=16, help='Number of hidden channels')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    return parser.parse_args()
