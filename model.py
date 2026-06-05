#the first part is the input embeddings
import torch
import torch.nn as nn
import math as maths

class InputEmbeddings(nn.Module): #nn.module is the parent class and the input embeddings is the child class
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__() #calling the parent class constructor
        self.d_model = d_model
        self.vocab_size = vocab_size #self. d model and the vocab size are normal class parameters
        self.embedding = nn.Embedding(vocab_size, d_model) #nn.Embedding is imported from nn.module

    def forward(self, x):
        return self.embedding(x) * maths.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
        #d_model because because the positional encoding follows the same dim as the input 
        #seq_len is the size max of the input sentence
        def __init__(self, d_model: int, seq_len: int, dropout: float):
         super().__init__()
         self.d_model = d_model
         self.seq_len = seq_len
         self.dropout = nn.Dropout(dropout) 

         #create a matrix of size seq_len x d_model
         pe = torch.zeros(seq_len, d_model)
         #we need to create a position vector of shape seq_len x 1
         position = torch.arange(0, seq_len).unsqueeze(1)
         div_term = torch.exp(torch.arange(0, d_model, 2) * -(maths.log(10000.0) / d_model))
         #apply th sine for even indices and cosine for odd indices
         pe[:, 0::2] = torch.sin(position * div_term)
         pe[:, 1::2] = torch.cos(position * div_term)
         pe = pe.unsqueeze(0) #add a batch dimension

         self.register_buffer('pe', pe) #register the positional encoding as a buffer so that it is not updated during training

        def forward(self, x):
            x = x + self.pe[:, :x.shape[1], :].requires_grad_(False) #add the positional encoding to the input
            return self.dropout(x)