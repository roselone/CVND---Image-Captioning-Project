import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn1 = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn1(features)
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        self.init_weights()
    
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0.01)
    
    def forward(self, features, captions):
        features = features.unsqueeze(1)    # batch_size x 1 x embed_size
        # remove <end> token from captions
        captions = captions[:, :-1]         # batch_size x (seq_len - 1)
        captions_em = self.embed(captions)  # batch_size x (seq_len - 1) x embed_size
        # construct input [image_feature : catptions ]
        inputs = torch.cat((features, captions_em), 1)  # batch_size x seq_len x embed_size
        lstm_out, _ = self.lstm(inputs)     # batch_size x seq_len x hidden_size
        out = self.fc(lstm_out)             # batch_size x seq_len x vocab_size
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        res = []
        for i in range(max_len):
            out, states = self.lstm(inputs, states)
            out = out.squeeze(1)   # 1 x hidden_size
            out = self.fc(out)     # 1 x vocab_size
            _, pred = out.max(1)
            res.append(pred.item())
            inputs = self.embed(pred).unsqueeze(1)
        return res