#the first part is the input embeddings
import torch
import torch.nn as nn
import math as maths

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model #self. d model and the vocab size are normal class parameters
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) #nn.Embedding is an imported fct from the nn.Module class

    def forward(self, x):
        return self.embedding(x) * maths.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
     #d_model because because the positional encoding follows the same dim as the input 
     #seq_len is the size max of the input sentence
    def __init__(self, d_model: int, seq_len: int, dropout: float): 
        super().__init__() #calling the parent class constructor
        self.d_model = d_model 
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) 

        #create a matrix of size seq_len * d_model
        pe = torch.zeros(seq_len, d_model)
        #we need to create a position vector of shape seq_len x 1
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(maths.log(10000.0) / d_model))
        #apply th sine for even indices and cosine for odd indices like the paper suggested
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        #add the positional encoding to the input
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
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 and B1 first linear layer that maps the input to a higher dimension
        self.dropout = nn.Dropout(dropout) #dropout layer to prevent overfitting
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 and B2 second linear layer that maps the output back to the original dimension

    def forward(self, x):
        # (Batch, Selq_Len, d_model) -> (Batch, Seq_Len, d_ff) -> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model #should be divisible by the number of heads
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / maths.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores #return the output of the attention and the attention scores for visualization also here we are multiplying the attention weights with the value to get the output of the attention

    def forward(self, q, k, v, mask):
        query = self.w_q(q) #(Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)
        key = self.w_k(k) #(Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)
        value = self.w_v(v) #(Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)

        #(Batch, Seq_Len, d_model) -> (Batch, Seq_Len, num_heads, d_k) -> (Batch, num_heads, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k) #(Batch, num_heads, Seq_Len, d_k) -> (Batch, Seq_Len, d_model)

        return self.w_o(x) 
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        #apply layer normalization to the input, then apply the sublayer, then apply dropout, and finally add the input to the output of the sublayer
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
         #basically we take the input x and call the forward method of multihead attention then we pass the output of the multihead attention to the first residual connection and add it to the input x
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feedforward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feedforward_block = feedforward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])  
       

    # ✅ FIX 1: forward() was nested inside __init__, moved out to class level
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feedforward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.projection(x), dim=-1)
    

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, num_heads: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    #create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    #positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # ✅ FIX 2: decoder_blocks loop was nested inside encoder_blocks loop — moved out
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    #encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    #create the transformer
    model = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    #initialize the weights of the transformer
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # ✅ FIX 3: was returning Transformer (the class) instead of model (the instance)
    return model