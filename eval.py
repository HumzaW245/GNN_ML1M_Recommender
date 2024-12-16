
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
  model = gnn.GNNRecommender(x_train.size(1), 16)
  model.to(device)
  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  ratings_train_tensor = torch.tensor(ratings_train['rating'].values, dtype=torch.float)



  for epoch in range(args.epochs):
      print('reached here. closing for test')
      assert False
      model.train()
      optimizer.zero_grad()
      out = model(x_train, edge_index_train)
      loss = criterion(out, ratings_train_tensor)
      loss.backward()
      optimizer.step()
      print(f'Epoch {epoch+1}, Loss: {loss.item()}')

  model.eval() 
  with torch.no_grad(): 
    test_out = model(x_test.to(device), edge_index_test.to(device)) 
    test_loss = criterion(test_out, ratings_test_tensor) 
    print(f'Epoch {epoch+1}, Test Loss: {test_loss.item()}')
    # Calculate accuracy 
    # Example threshold, adjust as necessary 
    threshold = args.accuracyTheshold 
    test_predictions = torch.round(test_out).cpu().numpy() 
    test_actuals = ratings_test_tensor.cpu().numpy() 
    accuracy = (test_predictions == test_actuals).mean() 
    print(f'Epoch {epoch+1}, Test Accuracy: {accuracy * 100:.2f}%')

  # def predict(user_id, item_id):
  #     user_idx = users.index[users['user_id'] == user_id].tolist()[0]
  #     item_idx = num_users + movies.index[movies['item_id'] == item_id].tolist()[0]
  #     with torch.no_grad():
  #         pred = model(x, edge_index)
  #         return pred[user_idx, item_idx].item()

  # def recommend(user_id, top_k=5):
  #     user_idx = users.index[users['user_id'] == user_id].tolist()[0]
  #     item_indices = [num_users + i for i in range(num_items)]
  #     predictions = []
  #     with torch.no_grad():
  #         pred = model(x, edge_index)
  #         for item_idx in item_indices:
  #             predictions.append((item_idx - num_users, pred[user_idx, item_idx].item()))
  #     predictions.sort(key=lambda x: x[1], reverse=True)
  #     recommended_items = [item[0] for item in predictions[:top_k]]
  #     return recommended_items

  # '''
  # Example: Predict rating for a specific user-item pair
  # user_id = 1  # Replace with the user_id you want to predict for
  # item_id = 10  # Replace with the item_id you want to predict for
  # predicted_rating = predict(user_id, item_id)
  # print(f'Predicted rating for user {user_id} and item {item_id}: {predicted_rating}')

  # # Example: Get top 5 recommendations for a user
  # user_id = 1  # Replace with the user_id you want recommendations for
  # recommended_items = recommend(user_id, top_k=5)
  # print(f'Top 5 recommendations for user {user_id}: {recommended_items}')

  # '''





'''
Executing Run (optional: with a custom config)
'''

#Use by running in command line e.g.: python evaluateSpurious.py --config_string "learning.learning_rate=0.001, learning.epochs=102, learning.train_batch_size=64, learning.finetune_backbones=False, printTraining=False"


if __name__ == '__main__':
    print('STARTING PROGRAM')
    args = config.parse_args()

    evaluate(args)