import torch

class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass


class Recall(Metrics):
    """
    Args:
         ats (int): @ to eval.
         rank_na (bool): whether to consider no answer.
    """
    def __init__(self, at=10):
        self.at = at
        self.n = 0
        self.n_correct = 0
        self.name = 'Recall@{}'.format(at)

    def reset(self):
        self.n = 0
        self.n_corrects = 0

    def update(self, predicts, batch):
        """
        Args:
            predicts (FloatTensor): with size (batch, n_samples).
            batch (dict): batch.
        """
        predicts = predicts.cpu()
        self.n += predicts.size(0)
        
        if self.at >= predicts.size(1):
            pred_idx = torch.argmax(predicts, dim=-1)
            label_idx = torch.argmax(batch['labels'], dim=-1)
            for i in range(predicts.size(0)):
                if pred_idx[i] == label_idx[i]:
                    self.n_corrects += 1
        else:
            label_idx = torch.argmax(batch['labels'], dim=-1)
            for i in range(predicts.size(0)):
                pred_idx = set(sorted(range(len(predicts[i])), key=lambda k: predicts[i][k], reverse=True)[:self.at])
                if label_idx[i].item() in pred_idx:
                    self.n_corrects += 1
            
    def get_score(self):
        return self.n_corrects / self.n

    def print_score(self):
        score = self.get_score()
        return '{:.2f}'.format(score)
