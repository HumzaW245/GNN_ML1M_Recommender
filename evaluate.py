import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
'''
#NEED TO CITE DATA SOURCE (MovieLens 1M Dataset https://grouplens.org/datasets/movielens/1m/

https://grouplens.org/datasets/movielens/?form=MG0AV3)

CITATION
================================================================================

To acknowledge use of the dataset in publications, please cite the following
paper:

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4,
Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

ACKNOWLEDGEMENTS
================================================================================

'''
def evaluate():
    # Load the data
    ratings = pd.read_csv('data/ratings.dat', sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
    movies = pd.read_csv('data/movies.dat', sep='::', names=['item_id', 'title', 'genres'], engine='python', encoding='latin-1')
    users = pd.read_csv('data/users.dat', sep='::', names=['user_id', 'gender', 'age', 'occupation', 'zip_code'], engine='python', encoding='latin-1')

    # Create user-item interaction matrix
    num_users = ratings.user_id.nunique()
    num_items = ratings.item_id.nunique()

    edge_index = torch.tensor([ratings.user_id.values, ratings.item_id.values], dtype=torch.long)

    # Create node features (user and item features)
    user_features = torch.tensor(users[['age', 'occupation']].values, dtype=torch.float)
    item_features = torch.zeros((num_items, user_features.size(1)))
    x = torch.cat([user_features, item_features], dim=0)

    class GNNRecommender(torch.nn.Module):
        def __init__(self, num_features, hidden_channels):
            super(GNNRecommender, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.fc = torch.nn.Linear(hidden_channels, 1)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.fc(x)
            return x

    model = GNNRecommender(x.size(1), 16)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ratings_tensor = torch.tensor(ratings['rating'].values, dtype=torch.float)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out, ratings_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

if __name__ == '__main__':
    evaluate()
