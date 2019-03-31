import torch
import pickle
import argparse
import logging
import matplotlib; matplotlib.use('Agg')
import seaborn as sns; sns.set()
import numpy as np

def main(args):
    # load data
    with open(args.data_path, "rb+") as fin:
        data = pickle.load(fin)
    idx = 75
    data = data.data[idx]

    # load embedding model
    with open(args.embed_path, "rb+") as fin:
        embed = pickle.load(fin)
        embedding = embed.vectors
    embedding_model = torch.nn.Embedding(embedding.size(0), embedding.size(1))
    embedding_model.weight = torch.nn.Parameter(embedding)
    key = np.array(list(embed.word_dict.keys()))

    # load model
    with open("../models/best/model.pkl", "rb+") as fin:
        model = torch.load(fin, map_location='cpu')
    lstm = LstmAttentionNet(303, 128, 1, 1, True)
    lstm.load_state_dict(model['model'])

    data = preprocess(data)
    batch = collate_fn([data])

    context = embedding_model(batch['context'])
    speaker = batch['speaker']
    context = torch.cat((context, speaker), dim=-1)
    context_lens = torch.LongTensor(batch['context_lens'])
    options = embedding_model(batch['options'])
    option_lens = torch.LongTensor(batch['option_lens'])

    logits, attention_map = lstm.forward(
        torch.cat((context, context), dim=0),
        context_lens,
        torch.cat((options, options), dim=0),
        option_lens)

    # plot the attention score
    x_label = key[batch['context'][0].data.numpy()]
    y_label = key[batch['options'][0, 0][-11:].data.numpy()]

    matplotlib.pyplot.figure(figsize=(15, 12))
    img = attention_map[0, 0, -11:].detach().cpu().numpy()
    ax = sns.heatmap(img, linewidths=0.05, cmap='YlGnBu', xticklabels=x_label, yticklabels=y_label)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=75, fontsize=12)
    figure = ax.get_figure()
    figure.savefig("attention_map.png")

def preprocess(data):
    positives = data['options'][:data['n_corrects']]
    negatives = data['options'][data['n_corrects']:]
    positive_ids = data['option_ids'][:data['n_corrects']]
    negative_ids = data['option_ids'][data['n_corrects']:]
    n_positive = len(positives)
    n_negative = len(negatives)
    context_padded_len = 300
    data['labels'] = [1] * n_positive + [0] * n_negative

    speaker = []
    context = []
    for i in range(len(data['context'])):
        if len(data['context'][i]) > context_padded_len:
            context += data['context'][i][:context_padded_len]
            speaker += [data['speaker'][i]] * context_padded_len
        else:
            context += data['context'][i]
            speaker += [data['speaker'][i]] * len(data['context'][i])

    data['speaker'] = speaker
    data['context'] = context
    return data

def collate_fn(datas):
    context_padded_len = 300
    option_padded_len = 50
    padding = 0

    batch = {}
    # collate lists
    batch['id'] = [data['id'] for data in datas]
    batch['speaker'] = [data['speaker'] for data in datas]
    batch['labels'] = torch.tensor([data['labels'] for data in datas])
    batch['option_ids'] = [data['option_ids'] for data in datas]

    # build tensor of context
    batch['context_lens'] = [min(len(data['context']), context_padded_len) for data in datas]
    padded_len = min(context_padded_len, max(batch['context_lens']))
    batch['context'] = torch.tensor(
        [pad_to_len(data['context'], padded_len, padding)
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
        [min(max(len(opt), 1), option_padded_len)
         for opt in data['options']]
        for data in datas]
    padded_len = min(
        option_padded_len,
        max(sum(batch['option_lens'], []))
    )
    batch['options'] = torch.tensor(
        [[pad_to_len(opt, padded_len, padding)
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
        attention_map = []
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
            attention_map.append(attention_opt)
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
        attention_map = torch.stack(attention_map, 0)
        return logits, attention_map

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to plot attention score map.")
    parser.add_argument('data_path', type=str,
                        help='Path to the data pickle file.')
    parser.add_argument('embed_path', type=str,
                        help='Path to the embedding pickle file.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)
