import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer

class CharLSTM(nn.Module):
    
    def __init__(self, char2int_dict, int2char_dict, n_steps=100, n_hidden=256, n_layers=2, bidirectional=False,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.D = 2 if bidirectional == True else 1
        
        # Creating character dictionaries
        self.int2char = int2char_dict
        self.char2int = char2int_dict
        
        ## Define the LSTM
        self.lstm = nn.LSTM(len(self.char2int), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True, bidirectional=bidirectional)
        
        ## Define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        ## Define the final, fully-connected output layer
        self.fc = nn.Linear(self.n_hidden*self.D, len(self.char2int))
        
        # Initialize the weights
        self.init_weights()
      
    
    def forward(self, x, hc):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hc`. '''
        
        ## Get x, and the new hidden state (h, c) from the lstm
        x, (h, c) = self.lstm(x, hc)
        
        ## Ppass x through the dropout layer
        x = self.dropout(x)
        
        # Stack up LSTM outputs using reshape
        x = x.reshape(x.size()[0]*x.size()[1], -1)
        
        ## Put x through the fully-connected layer
        x = self.fc(x)
        
        # Return x and the hidden state (h, c)
        return x, (h, c)
    
    def init_weights(self):
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)
        
    #def init_hidden(self, n_seqs):
    def init_hidden(self, dims):
        ''' Initializes hidden state '''
        # # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # # initialized to zero, for hidden state and cell state of LSTM
        # weight = next(self.parameters()).data
        # return (weight.new(self.D*self.n_layers, n_seqs, self.n_hidden).zero_(),
        dims = (self.D*self.n_layers, *dims, self.n_hidden) # dims should be a tuple type
        return (torch.zeros(dims), torch.zeros(dims))

def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    
    batch_size = n_seqs * n_steps
    n_batches = len(arr)//batch_size
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size]
    
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, -1))
    
    for n in range(0, arr.shape[1], n_steps):
        
        # The features
        x = arr[:, n:n+n_steps]
        
        # The targets, shifted by one
        y = torch.zeros_like(x)
        
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+n_steps]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

def load_from_folder(data_folder_path, tokenizer):
    train_content_for_training, train_content_for_vocab = extract_tweet(os.path.join(data_folder_path, "train.txt"), tokenizer)
    val_content_for_training, val_content_for_vocab = extract_tweet(os.path.join(data_folder_path, "valid.txt"), tokenizer)
    test_content_for_training, test_content_for_vocab = extract_tweet(os.path.join(data_folder_path, "test.txt"), tokenizer)
    content_for_training = train_content_for_training + val_content_for_training + test_content_for_training
    content_for_vocab = train_content_for_vocab + val_content_for_vocab + test_content_for_vocab

    chars = tuple(set(content_for_vocab))
    assert(" " in chars)
    int2char_old = dict(enumerate(chars))
    int2char = {k+1: v for k, v in int2char_old.items()}
    int2char[0] = "[PAD]"
    int2char[len(int2char)] = "[CLS]"
    int2char[len(int2char)] = "[SEP]"
    int2char[len(int2char)] = "[ENT-PER]"
    int2char[len(int2char)] = "[ENT-ORG]"
    int2char[len(int2char)] = "[ENT-LOC]"
    int2char[len(int2char)] = "[ENT-MISC]"
    char2int = {ch: ii for ii, ch in int2char.items()}
    encoded = []
    for token in content_for_training:
        if token in char2int:
            char_ids = [char2int[token]]
        else:
            chars = list(token)
            char_ids = [char2int.get(c, 0) for c in chars]
        encoded += char_ids
    encoded = torch.as_tensor(encoded)
    return char2int, int2char, encoded

def extract_tweet(file_path, tokenizer):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        content_for_training = []
        content_for_vocab = []
        for line in lines:
            if line.startswith("IMGID"):
                content_for_training.append("[CLS]") # Start of a tweet
            elif line == '\n':
                content_for_training.append("[SEP]") # End of a tweet
            else:
                content_for_vocab.append(line.split('\t')[0])
                tokens = tokenizer.tokenize(line.split('\t')[0])
                content_for_training.extend(tokens)
        content_for_vocab = ' '.join(content_for_vocab)
    return content_for_training, content_for_vocab

def train(net, data, epochs=10, n_seqs=10, n_steps=50, lr=0.001, clip=5, val_frac=0.1, cuda=False, print_every=10):
    ''' Training a network 
    
        Arguments
        ---------
        
        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        n_seqs: Number of mini-sequences per mini-batch, aka batch size
        n_steps: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        cuda: Train with CUDA on a GPU
        print_every: Number of steps for printing training and validation loss
    
    '''
    
    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    
    criterion = nn.CrossEntropyLoss()
    
    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    if cuda:
        net.cuda()
    
    counter = 0
    n_chars = len(net.char2int)
    
    for e in range(epochs):
        
        h = net.init_hidden((n_seqs,))
        
        for x, y in get_batches(data, n_seqs, n_steps):
            
            counter += 1
            
            # One-hot encode our data and make them Torch tensors
            inputs, targets = F.one_hot(x, n_chars), y
            inputs = inputs.to(torch.float32)
            
            if cuda:
                inputs, targets, h = inputs.cuda(), targets.cuda(), tuple([each.cuda() for each in h])

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            net.zero_grad()
            
            output, h = net.forward(inputs, h)
            
            loss = criterion(output, targets.view(n_seqs*n_steps).type(torch.cuda.LongTensor))

            loss.backward()
            
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)

            opt.step()
            
            if counter % print_every == 0:
                
                # Get validation loss
                val_h = net.init_hidden((n_seqs,))
                val_losses = []
                
                for x, y in get_batches(val_data, n_seqs, n_steps):
                    
                    # One-hot encode our data and make them Torch tensors
                    x = F.one_hot(x, n_chars)
                    inputs, targets = x.to(torch.float32), y
                    
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])
                    
                    if cuda:
                        inputs, targets, val_h = inputs.cuda(), targets.cuda(), tuple([each.cuda() for each in val_h])

                    output, val_h = net.forward(inputs, val_h)

                    val_loss = criterion(output, targets.view(n_seqs*n_steps).type(torch.cuda.LongTensor))
                
                    val_losses.append(val_loss.item())
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))
          
if __name__ == "__main__":
    data_folder_path = "data/NER_data/twitter2015"
    local_cache_path = "/home/yixin/workspace/huggingface/"
    lm_name = "vinai/bertweet-base"
    n_seq, n_char = 128, 100

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(local_cache_path, lm_name))
    char2int, int2char, encoded = load_from_folder(data_folder_path, tokenizer)
    net = CharLSTM(char2int, int2char, n_hidden=512, n_layers=2, bidirectional=True)
    train(net, encoded, epochs=10, n_seqs=n_seq, n_steps=n_char, lr=0.001, cuda=True, print_every=10)

    torch.save(net.state_dict(), "char_lstm.pth")
    torch.save([char2int, int2char], "char_vocab.pkl")