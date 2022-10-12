import torch
from torch.utils.data import Dataset, DataLoader
from config import TEXT_COL, TARGET_COL, MAX_LEN, BATCH_SIZE

class VersesDataset(Dataset):
    def __init__(self, verses, targets, tokenizer, max_len):
        self.verses = verses
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.verses)
    
    def __getitem__(self, item):
        verse = str(self.verses[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            verse,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
          'text': verse,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }

    
def create_data_set(df, tokenizer, max_len=MAX_LEN):
    ds = VersesDataset(
        verses=df[TEXT_COL].to_numpy(),
        targets=df[TARGET_COL].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return ds
    
def create_data_loader(df, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE):
    ds = VersesDataset(
        verses=df[TEXT_COL].to_numpy(),
        targets=df[TARGET_COL].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size)