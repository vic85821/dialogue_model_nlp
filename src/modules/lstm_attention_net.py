import torch

class LstmAttentionNet(torch.nn.Module):
    def __init__(self, dim_embeddings, hidden_size, output_size, hidden_layer, bidirectional):
        '''
        
        Args:
            hidden_size: hidden_layer output feature size
            output_size: this network output size
            hidden_layer: the number of the hidden_layer in the RNN
            bidirectional: use bidirection or not in the RNN
        '''
        super(LstmAttentionNet, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # the output feature of the RNN: (hidden feature size x hidden layer number) 
        self.feature = self.hidden_size * hidden_layer
        
        # if use the bidirection, the feature number need to multiply 2
        if bidirectional == True:
            self.feature *= 2
            
        # first lstm used to extract feature from embeding word of the context and option
        self.lstm1 = torch.nn.LSTM(dim_embeddings, self.hidden_size, hidden_layer, bidirectional=bidirectional)
        # second lstm used to extract feature from attentioned option
        self.lstm2 = torch.nn.LSTM(self.feature*3, self.hidden_size, hidden_layer, bidirectional=bidirectional)
        # the mlp is used to calculate the attention score
        self.linear_opt = torch.nn.Linear(self.feature, self.feature)
        
        # the mlp is used for comparation between the context feature and the option feature
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.feature*4, self.hidden_size),
            torch.nn.BatchNorm1d(self.hidden_size),
            torch.nn.ReLU(),
            
            torch.nn.Linear(self.hidden_size, self.output_size),
            torch.nn.Sigmoid()
        )
        
        # pad the option embeding with more three values (speaker one-hot encode)
        self.pad = torch.nn.ConstantPad1d((0, 3), 0)
        self.softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self, context, context_lens, options, option_lens):
        '''
        
        Args:
            context: tensor of the context, (batch, cont_seq, embeding_size)
            context_lens: tensor of the context len, (batch)
            options: tensor of the several options, (batch, opt_num, opt_seq, embedding_size)
            option_lens: tensor of the option len, (batch, opt_num)
        '''
        # output: (cont_seq, batch, hidden), h_n: (num_layer, batch, hidden)
        context, (h_n, c_n) = self.lstm1(context.transpose(1, 0))
        # output_cont: (batch, hidden)
        mean_cont = torch.mean(context, dim=0)
        output_cont = torch.cat((context[-1], mean_cont), dim=-1)
        
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # option: (opt_seq, batch, hidden), h_n: (num_layer, batch, hidden)
            option, (h_n, c_n) = self.lstm1(self.pad(option.transpose(1, 0)))
            # option_weighted: (opt_seq, batch, hidden)
            option_weighted = self.linear_opt(option)
            
            # _: (batch, opt_seq, cont_seq) <-- (batch, opt_seq, hidden) * (batch, hidden, cont_seq)
            _ = torch.bmm(option_weighted.transpose(1, 0), context.transpose(1, 0).transpose(2, 1)) 
            # softmax on cont_seq dimention
            attention_opt = self.softmax(_)
            # (batch, opt_seq, hidden) <-- (batch, opt_seq, cont_seq) * (batch, cont_seq, hidden)
            mix_opt = torch.bmm(
                attention_opt,
                context.transpose(1, 0)
            )
            # mix_opt: (opt_seq, batch, cont_seq)
            mix_opt = mix_opt.transpose(1, 0) * option
            
            # attention_cont: (batch, cont_seq)
            attention_cont = torch.max(_, dim=1)[0]
            # attention_cont: (batch, 1, cont_seq)
            attention_cont = self.softmax(attention_cont).unsqueeze(1)
            # mix_cont: (batch, 1, hidden)
            mix_cont = torch.bmm(
                attention_cont,
                context.transpose(1, 0)
            )
            mix_cont = mix_cont.transpose(1, 0) * option
            
            # mix: (opt_seq, batch, hidden*3)
            mix = torch.cat((option, mix_opt, mix_cont), dim=2)
            
            # output_opt: (batch, hidden)
            option, h_n = self.lstm2(mix)
            mean_opt = torch.mean(option, dim=0)
            output_opt = torch.cat((option[-1], mean_opt), dim=-1)
            
            # logit: (batch, 1)
            logit = self.mlp(torch.cat((output_cont, output_opt), dim=-1)).squeeze(1)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits