import torch
from base_predictor import BasePredictor
from modules import *

class Predictor(BasePredictor):
    """

    Args:
        embedding: matrix for embeding (word size x embedding size)
        arch: model name
        loss: loss name
        hidden_size: the hidden feature size of the RNN network
        output_size: the output size of the model
        hidden_layer: the hidden layer num of the RNN network
        bidirectional: use bidirectional in the RNN or not
        kwargs:
            batch_size
            max_epochs
            learning_rate
            n_workers
    """

    def __init__(self, embedding, arch=None, loss='BCELoss',
                 hidden_size=64, output_size=1, hidden_layer=1, bidirectional=False, **kwargs):
       
        super(Predictor, self).__init__(**kwargs)
        
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)

        self.model = {
            'LstmNet': LstmNet(embedding.size(1)+3, hidden_size, output_size, hidden_layer, bidirectional),
            'LstmAttentionNet': LstmAttentionNet(embedding.size(1)+3, hidden_size, output_size, hidden_layer, bidirectional),
            'GruNet': GruNet(embedding.size(1)+3, hidden_size, output_size, hidden_layer, bidirectional),
            'GruAttentionNet': GruAttentionNet(embedding.size(1)+3, hidden_size, output_size, hidden_layer, bidirectional)
        }[arch]
        
        self.model = self.model.to(self.device)
        self.embedding = self.embedding.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)        
        
        self.loss = {
            'BCELoss': torch.nn.BCELoss()
        }[loss]

    def _run_iter(self, batch, training):
        with torch.no_grad():
            context = self.embedding(batch['context'].to(self.device))
            speaker = batch['speaker'].to(self.device)
            context = torch.cat((context, speaker), dim=-1)
            context_lens = torch.LongTensor(batch['context_lens'])
            options = self.embedding(batch['options'].to(self.device))
            option_lens = torch.LongTensor(batch['option_lens'])
        
        logits = self.model.forward(
            context.to(self.device),
            context_lens.to(self.device),
            options.to(self.device),
            option_lens.to(self.device))
        
        loss = self.loss(logits, batch['labels'].float().to(self.device))
        return logits, loss

    def _predict_batch(self, batch):
        context = self.embedding(batch['context'].to(self.device))
        speaker = batch['speaker'].to(self.device)
        context = torch.cat((context, speaker), dim=-1)
        context_lens = torch.LongTensor(batch['context_lens'])
        options = self.embedding(batch['options'].to(self.device))
        option_lens = torch.LongTensor(batch['option_lens'])
        
        logits = self.model.forward(
            context.to(self.device),
            context_lens.to(self.device),
            options.to(self.device),
            option_lens.to(self.device))
        
        return logits
