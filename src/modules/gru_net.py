import torch

class GruNet(torch.nn.Module):
    def __init__(self, dim_embeddings, hidden_size, output_size, hidden_layer, bidirectional):
        super(GruNet, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.gru = torch.nn.GRU(dim_embeddings, self.hidden_size, 1)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size*2, self.hidden_size),
            torch.nn.BatchNorm1d(self.hidden_size),
            torch.nn.ReLU(),
            
            torch.nn.Linear(self.hidden_size, self.output_size),
            torch.nn.Sigmoid()
        )

        self.pad = torch.nn.Sequential(
            torch.nn.ConstantPad1d((0, 3), 0)
        )
        
    def forward(self, context, context_lens, options, option_lens):
        output, _ = self.gru(context.transpose(1, 0))
        context = output[-1]
        
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            output, _ = self.gru(self.pad(option.transpose(1, 0)))
            option = output[-1]
            logit = self.mlp(torch.cat((context, option), dim=-1)).squeeze(1)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits