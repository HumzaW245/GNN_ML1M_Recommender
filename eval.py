
import GNNRecommender as gnn
import torch.nn as nn
import torch
import numpy as np

import config
import data.data as pipeLine
import wandb
import argparse

def evaluate(args):
  
  wandb.login()
  #------------------------------------------------------>INITIALIZING WANDB PROJECT NAME AND NAME OF RUN <--------------------------------------------------
  wandb.init(project="GNN_Project_MAT6495", name='MovieLens1M: ' + args.wandbNameSuffix)

  #Making sure config dictionary is update to have values expected
  print(f'\n\n\nThe configuration for this run is as follows: \n {args} \n\n\n')


  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print(device)

  x_train, edge_index_train, ratings_train, x_test, edge_index_test, ratings_test = pipeLine.get_movielens1m_train_test(args)


  #  model = gnnModel(9746, 16)
  model = gnn.GNNRecommender(x_train.size(1), args.hidden_channels)
  model.to(device)
  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  ratings_train_tensor = torch.tensor(ratings_train['rating'].values, dtype=torch.float)

 

  for epoch in range(args.epochs):

      model.train()  
      optimizer.zero_grad()
      out = model(x_train.to(device), edge_index_train.to(device))
      loss = criterion(out.to(device), ratings_train_tensor.to(device))
      loss.backward()
      optimizer.step()
      print(f'Epoch {epoch+1}, Loss: {loss.item()}')

  model.eval() 
  with torch.no_grad(): 
    test_out = model(x_test.to(device), edge_index_test.to(device)) 
    test_loss = criterion(test_out.to(device), ratings_test_tensor.to(device)) 
    print(f'Epoch {epoch+1}, Test Loss: {test_loss.item()}')
    # Calculate accuracy 
    # Example threshold, adjust as necessary 
    threshold = args.accuracyTheshold 
    test_predictions = torch.round(test_out).cpu().numpy() 
    test_actuals = ratings_test_tensor.cpu().numpy() 
    accuracy = (test_predictions == test_actuals).mean() 
    print(f'Epoch {epoch+1}, Test Accuracy: {accuracy * 100:.2f}%')




'''
Executing Run (optional: with a custom config)
'''

#Use by running in command line e.g.: python evaluateSpurious.py --config_string "learning.learning_rate=0.001, learning.epochs=102, learning.train_batch_size=64, learning.finetune_backbones=False, printTraining=False"


if __name__ == '__main__':
    print('STARTING PROGRAM')
    args = config.parse_args()

    evaluate(args)