import random
import torch
from torch.utils.data import Dataset
import numpy as np

class DialogDataset(Dataset):
    """
    Args:
        data (list): List of samples.
        padding (int): Index used to pad sequences to the same length.
        n_negative (int): Number of false options used as negative samples to
            train. Set to -1 to use all false options.
        n_positive (int): Number of true options used as positive samples to
            train. Set to -1 to use all true options.
        shuffle (bool): Do not shuffle options when sampling.
            **SHOULD BE FALSE WHEN TESTING**
    """
    
    def __init__(self, data, padding=0,
                 n_negative=-1, n_positive=-1,
                 context_padded_len=300, option_padded_len=50, 
                 shuffle=False):
        self.data = data
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.context_padded_len = context_padded_len
        self.option_padded_len = option_padded_len
        self.padding = padding
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = dict(self.data[index])
        positives = data['options'][:data['n_corrects']]
        negatives = data['options'][data['n_corrects']:]
        positive_ids = data['option_ids'][:data['n_corrects']]
        negative_ids = data['option_ids'][data['n_corrects']:]

        if self.n_positive == -1:
            n_positive = len(positives)
        if self.n_negative == -1:
            n_negative = len(negatives)
        else:
            n_positive = min(len(positives), self.n_positive)
            n_negative = min(len(negatives), self.n_negative)

        if self.shuffle == True:
            positive_indices = random.sample(list(np.arange(len(positives))), k=n_positive)
            negative_indices = random.sample(list(np.arange(len(negatives))), k=n_negative)

            # collect sampled options
            data['options'] = (
                [positives[i] for i in positive_indices]
                + [negatives[i] for i in negative_indices]
            )
            data['option_ids'] = (
                [positive_ids[i] for i in positive_indices]
                + [negative_ids[i] for i in negative_indices]
            )
            data['labels'] = [1] * n_positive + [0] * n_negative
        
            lists = list(zip(data['options'], data['option_ids'], data['labels']))
            random.shuffle(lists)
            data['options'], data['option_ids'], data['labels'] = zip(*lists)
        else:
            data['labels'] = [1] * n_positive + [0] * n_negative
        
        speaker = []
        context = []
        for i in range(len(data['context'])):
            if len(data['context'][i]) > self.context_padded_len:
                context += data['context'][i][:self.context_padded_len]
                speaker += [data['speaker'][i]] * self.context_padded_len
            else:
                context += data['context'][i]
                speaker += [data['speaker'][i]] * len(data['context'][i])
        
        data['speaker'] = speaker
        data['context'] = context
        
        return data

    def collate_fn(self, datas):
        batch = {}
        # collate lists
        batch['id'] = [data['id'] for data in datas]
        batch['speaker'] = [data['speaker'] for data in datas]
        batch['labels'] = torch.tensor([data['labels'] for data in datas])
        batch['option_ids'] = [data['option_ids'] for data in datas]
       
        # build tensor of context
        batch['context_lens'] = [min(len(data['context']), self.context_padded_len) for data in datas]
        padded_len = min(self.context_padded_len, max(batch['context_lens']))
        batch['context'] = torch.tensor(
            [pad_to_len(data['context'], padded_len, self.padding)
             for data in datas], dtype=torch.int64
        )
        
        # build tensor of speaker
        speaker = [pad_to_len(data['speaker'], padded_len, data['speaker'][-1]) for data in datas]
        batch['speaker'] = torch.zeros(batch['context'].size(0), batch['context'].size(1), 3)
        for i in range(batch['speaker'].size(0)):
            for j in range(batch['speaker'].size(1)):
                batch['speaker'][i][j][speaker[i]] = 1.0
        
        # build tensor of options
        batch['option_lens'] = [
            [min(max(len(opt), 1), self.option_padded_len)
             for opt in data['options']]
            for data in datas]
        padded_len = min(
            self.option_padded_len,
            max(sum(batch['option_lens'], []))
        )
        batch['options'] = torch.tensor(
            [[pad_to_len(opt, padded_len, self.padding)
              for opt in data['options']]
             for data in datas], dtype=torch.int64
        )
        return batch


def pad_to_len(arr, padded_len, padding=0):
    """ Pad `arr` to `padded_len` with padding if `len(arr) < padded_len`.
    If `len(arr) > padded_len`, truncate arr to `padded_len`.
    Example:
        pad_to_len([1, 2, 3], 5, -1) == [1, 2, 3, -1, -1]
        pad_to_len([1, 2, 3, 4, 5, 6], 5, -1) == [1, 2, 3, 4, 5]
    Args:
        arr (list): List of int.
        padded_len (int)
        padding (int): Integer used to pad.
    """
    arr_len = len(arr)
    if arr_len < padded_len:
        arr = list(np.pad(arr, (padded_len-arr_len, 0), 'constant', constant_values=(padding)))
    elif len(arr) > padded_len:
        arr = arr[-padded_len:]
    return arr
