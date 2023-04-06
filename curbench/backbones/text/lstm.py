import torch
import torch.nn as nn


# LSTM-based attentional base-line

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, num_labels, embedding_dim=300, hidden_dim=512, 
                 n_layers=3, use_bidirectional=False, use_dropout=False):
        super().__init__()
        self.num_labels = num_labels
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           batch_first=True, bidirectional=use_bidirectional,
                           dropout=0.5 if use_dropout else 0.)
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, self.num_labels)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

        self.attention_size = 1
        self.num_direction = 2
        self.w_omega = Variable(torch.zeros(self.hidden_dim * self.num_direction, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))

    def attention_net(self, lstm_output):
        #lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        """
        print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)
        print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)
        print(attn_tanh.size())  (squence_length * batch_size, attention_size)
        print(attn_hidden_layer.size())  (squence_length * batch_size, 1)
        print(exps.size())  (batch_size, squence_length)
        print(alphas.size()) (batch_size, squence_length)
        print(alphas_reshape.size()) = (batch_size, squence_length, 1)
        print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)
        print(attn_output.size()) = (batch_size, hidden_size*layer_size)
        """

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_dim*self.num_direction])
        # M = tanh(H)
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # alpha = softmax(omega.T*M)
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, lstm_output.size()[0]])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, lstm_output.size()[0], 1])
        state = lstm_output.permute(1, 0, 2)
        # r = H*alpha.T
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, input_ids, **kwargs):
        # self.rnn.flatten_parameters()
        embedded = self.embedding(input_ids)
        output, (hidden, cell) = self.rnn(embedded)
        last_hidden = self.dropout(hidden[-1, :, :])
        attn_output = self.attention_net(last_hidden)
        logits = self.fc(attn_output)
        return logits, hidden, cell

class LSTM(nn.Module):
    def __init__(self, vocab_size, num_labels, embedding_dim=300, hidden_dim=128, 
                 n_layers=3, use_bidirectional=False, use_dropout=False):
        super().__init__()
        self.num_labels = num_labels
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           batch_first=True, bidirectional=use_bidirectional,
                           dropout=0.5 if use_dropout else 0.)
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, self.num_labels)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)


    def forward(self, input_ids, **kwargs):
        # self.rnn.flatten_parameters()
        embedded = self.embedding(input_ids)
        output, (hidden, cell) = self.rnn(embedded)
        last_hidden = self.dropout(hidden[-1, :, :])
        logits = self.fc(last_hidden)
        return logits, hidden, cell
