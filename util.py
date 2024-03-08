import torch

class Translation_Dataset(torch.utils.data.Dataset):
    def __init__(self,source,target,tok):
        self.src = []
        with open(source,'r') as file:
            self.src = [l for l in file]
        self.tgt = []
        with open(target,'r') as file:
            self.tgt = [l for l in file]
        self.tok = tok
    
    def __len__(self):
        return len(self.src)

    def __getitem__(self,idx):
        return self.tok(self.src[idx], text_target=self.tgt[idx], max_length=128, truncation=True)