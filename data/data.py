
# Split the ratings into train and test sets 
from sklearn.model_selection import train_test_split


import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse

class CustomDataset(Dataset):
    def __init__(self, ratings, users, movies):
        self.ratings = ratings
        self.users = users
        self.movies = movies
        self.edge_index = torch.tensor([self.ratings.user_id.values, self.ratings.item_id.values], dtype=torch.long)
        
        num_users = self.ratings.user_id.nunique()
        num_items = self.ratings.item_id.nunique()

        # Create node features (user and item features)
        user_features = torch.tensor(self.users[['age', 'occupation']].values, dtype=torch.float)
        item_features = torch.zeros((num_items, user_features.size(1)))
        self.data = torch.cat([user_features, item_features], dim=0)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[:, idx], self.edge_index[:, idx], self.ratings[idx]

def get_loader(data, batch_size=32, shuffle=True, drop_last = True):
    loader = torch.utils.data.DataLoader(data,
                                            batch_size=b_size,
                                            shuffle=True,
                                            drop_last = True)
    return loader



def get_movielens1m_loaders(args):
    # DATA
    ratings = pd.read_csv('data/ratings.dat', sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
    movies = pd.read_csv('data/movies.dat', sep='::', names=['item_id', 'title', 'genres'], engine='python', encoding='latin-1')
    users = pd.read_csv('data/users.dat', sep='::', names=['user_id', 'gender', 'age', 'occupation', 'zip_code'], engine='python', encoding='latin-1')


    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

    train_data = CustomDataset(train_ratings, users, movies)
    test_data = CustomDataset(test_ratings, users, movies)

    '''
    Send to dataloader
    '''
    train_loader = get_loader(train_data, args.batch_size, shuffle = True, drop_last = True)
    test_loader = get_loader(train_data, args.batch_size, shuffle = True, drop_last = True)
    return train_loader, test_loader



def get_movielens1m_train_test(args):
    # DATA
    ratings = pd.read_csv('data/ratings.dat', sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
    movies = pd.read_csv('data/movies.dat', sep='::', names=['item_id', 'title', 'genres'], engine='python', encoding='latin-1')
    users = pd.read_csv('data/users.dat', sep='::', names=['user_id', 'gender', 'age', 'occupation', 'zip_code'], engine='python', encoding='latin-1')


    ratings_train, ratings_test = train_test_split(ratings, test_size=0.2, random_state=42)
    # Filter users and movies for train and test sets 
    users_train = users[users['user_id'].isin(ratings_train['user_id'].unique())] 
    movies_train = movies[movies['item_id'].isin(ratings_train['item_id'].unique())] 
    users_test = users[users['user_id'].isin(ratings_test['user_id'].unique())] 
    movies_test = movies[movies['item_id'].isin(ratings_test['item_id'].unique())]
    
    '''
    Process train data
    '''

    num_users_train = ratings_train.user_id.nunique()
    num_items_train = ratings_train.item_id.nunique()

    edge_index_train = torch.tensor([ratings_train.user_id.values, ratings_train.item_id.values], dtype=torch.long)

    # Create node features (user and item features)
    user_features_train = torch.tensor(users_train[['age', 'occupation']].values, dtype=torch.float)
    item_features_train = torch.zeros((num_items_train , user_features_train.size(1)))
    x_train = torch.cat([user_features_train, item_features_train], dim=0)


    '''
    Process test data
    '''
    
    num_users_test = ratings_test.user_id.nunique()
    num_items_test = ratings_test.item_id.nunique()

    edge_index_test = torch.tensor([ratings_test.user_id.values, ratings_test.item_id.values], dtype=torch.long)

    # Create node features (user and item features)
    user_features_test = torch.tensor(users_test[['age', 'occupation']].values, dtype=torch.float)
    item_features_test = torch.zeros((num_items_test , user_features_test.size(1)))
    x_test = torch.cat([user_features_test, item_features_test], dim=0)
    return x_train, edge_index_train, ratings_train, x_test, edge_index_test, ratings_test