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
        

class LayerNorm(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #alpha is a learnable parameter that is multiplied to the normalized output
        self.bias = nn.Parameter(torch.zeros(1)) #bias is a learnable parameter that is added to the normalized output

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) #calculate the mean of the input
        std = x.std(-1, keepdim=True) #calculate the standard deviation of the input
        return self.alpha * (x - mean) / (std + self.eps) + self.bias #return the normalized output multiplied by alpha and added to bias
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model : int, d_ff : int, dropout : float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 and B1 first linear layer that maps the input to a higher dimension
        self.dropout = nn.Dropout(dropout) #dropout layer to prevent overfitting
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 and B2 second linear layer that maps the output back to the original dimension

    def forward(self, x):
        # (Batch, Selq_Len, d_model) -> (Batch, Seq_Len, d_ff) -> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) #linear1, then relu, then dropout, and finally linear2

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float):
        super().__init__()
        self.d_model = d_model #should be divisible by the number of heads
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads #dimension of each head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / maths.sqrt(d_k) #calculate the attention scores
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) #mask the attention scores where the mask is 0
        attention_scores = attention_scores.softmax(dim=-1) #apply softmax to get the attention weights
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores #return the output of the attention and the attention scores for visualization also here we are multiplying the attention weights with the value to get the output of the attention

    def forward(self, q, k, v, mask):
        query = self.w_q(q) #(Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)
        key = self.w_k(k) #(Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)
        value = self.w_v(v) #(Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)

        #(Batch, Seq_Len, d_model) -> (Batch, Seq_Len, num_heads, d_k) -> (Batch, num_heads, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k) #(Batch, num_heads, Seq_Len, d_k) -> (Batch, Seq_Len, d_model)

        #batch, seq_len, d_model -> batch, seq_len, d_model
        return self.w_o(x) 
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) #apply layer normalization to the input, then apply the sublayer, then apply dropout, and finally add the input to the output of the sublayer
    
class EncoderLayer(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) #two residual connections one for the self attention block and one for the feed forward block

    def forward(self,x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        #basically we take the input x and call the forward method of multihead attention then we pass the output of the multihead attention to the first residual connection and add it to the input x
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
    class Encoder(nn.Module):
        def __init__(self, layers : nn.ModuleList):
            super().__init__()
            self.layers = layers
            self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)