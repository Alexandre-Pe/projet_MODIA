import argparse
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader
from model import NCF 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Ratings_Dataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index()
        user_list = df['user_id'].unique()
        item_list = df['recipe_id'].unique()
        self.user2id = {w: i for i, w in enumerate(user_list)}
        self.item2id = {w: i for i, w in enumerate(item_list)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = self.user2id[self.df['user_id'][idx]]
        user = torch.tensor(user, dtype=torch.long)
        item = self.item2id[self.df['recipe_id'][idx]]
        item = torch.tensor(item, dtype=torch.long)
        rating = torch.tensor(self.df['rating'][idx], dtype=torch.float)
        return user, item, rating

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, default='weight.pth', help='Path to the weights')
    parser.add_argument('--test_path', type=str, default='test_script.csv', help='Path to a test file')
    args = parser.parse_args()

    interactions_train = pd.read_csv("data/interactions_train.csv")
    interactions_test = pd.read_csv(args.test_path)
    testloader = DataLoader(Ratings_Dataset(interactions_test), batch_size=64, num_workers=2)

    n_user = interactions_train.user_id.nunique()
    n_items = interactions_train.recipe_id.nunique()

    model = NCF(n_user, n_items).to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=torch.device(device)))  
    model.eval()

    users, items, r = next(iter(testloader))

    y = model(users, items)*5
    print("ratings", r.data)
    print("predictions:", y.flatten().data)