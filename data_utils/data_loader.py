import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class data_set(Dataset):

    def __init__(self, data,config=None):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx

    def collate_fn(self, data):
        label = torch.tensor([item[0]['label'] for item in data])
        input_ids = torch.cat([item[0]['features']['input_ids'] for item in data],dim=0)
        token_type_ids = torch.cat([item[0]['features']['token_type_ids'] for item in data],dim=0)
        attention_mask = torch.cat([item[0]['features']['attention_mask'] for item in data],dim=0)
        return (
            label,
            input_ids,
            token_type_ids,
            attention_mask
        )

def get_data_loader(config, data, shuffle = False, drop_last = False, batch_size = None):

    dataset = data_set(data, config)

    if batch_size == None:
        batch_size = min(config.batch_size, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader

class plus_data_set(Dataset):

    def __init__(self, data,config=None):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx

    def collate_fn(self, data):
        label = torch.tensor([item[0]['label'] for item in data])
        input_ids = torch.cat([item[0]['features']['input_ids'] for item in data],dim=0)
        token_type_ids = torch.cat([item[0]['features']['token_type_ids'] for item in data],dim=0)
        attention_mask = torch.cat([item[0]['features']['attention_mask'] for item in data],dim=0)
        reason_input_ids = torch.cat([item[0]['reason_features']['input_ids'] for item in data],dim=0)
        reason_token_type_ids = torch.cat([item[0]['reason_features']['token_type_ids'] for item in data], dim=0)
        reason_attention_mask = torch.cat([item[0]['reason_features']['attention_mask'] for item in data], dim=0)
        return (
            label,
            input_ids,
            token_type_ids,
            attention_mask,
            reason_input_ids,
            reason_token_type_ids,
            reason_attention_mask
        )

def get_plus_data_loader(config, data, shuffle = False, drop_last = False, batch_size = None):

    dataset = plus_data_set(data, config)

    if batch_size == None:
        batch_size = min(config.batch_size, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader


class test_data_set(Dataset):

    def __init__(self, data,config=None):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx

    def collate_fn(self, data):
        label = torch.tensor([item[0]['label'] for item in data])
        input_ids = torch.cat([item[0]['text_features']['input_ids'] for item in data],dim=0)
        token_type_ids = torch.cat([item[0]['text_features']['token_type_ids'] for item in data],dim=0)
        attention_mask = torch.cat([item[0]['text_features']['attention_mask'] for item in data],dim=0)
        reason_input_ids = torch.cat([item[0]['image_features']['input_ids'] for item in data],dim=0)
        reason_token_type_ids = torch.cat([item[0]['image_features']['token_type_ids'] for item in data], dim=0)
        reason_attention_mask = torch.cat([item[0]['image_features']['attention_mask'] for item in data], dim=0)
        return (
            label,
            input_ids,
            token_type_ids,
            attention_mask,
            reason_input_ids,
            reason_token_type_ids,
            reason_attention_mask
        )

def get_data_test_loader(config, data, shuffle = False, drop_last = False, batch_size = None):

    dataset = test_data_set(data, config)

    if batch_size == None:
        batch_size = min(config.batch_size, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader