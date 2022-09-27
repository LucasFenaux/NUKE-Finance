import math
import torch.nn as nn
import torch
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class NormalizationEmbeddingLayer(nn.Module):
    def __init__(self):
        super(NormalizationEmbeddingLayer, self).__init__()

    def forward(self, x):
        normalized_x = []
        for i in range(x.size()[0]):
            min_val = x[i].min()
            shifted_x = x[i] - min_val
            max_val = shifted_x.max()
            normalized_x.append((shifted_x/max_val).unsqueeze(0))
        return torch.cat(normalized_x, dim=0)


class StockTransformer(nn.Module):
    def __init__(self, d_model: int = 4, nhead: int = 1, batch_first: bool = True, dropout_rate: float = 0.1,
                 max_len: int = 5000):
        super(StockTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, batch_first)
        self.positional_encoding = PositionalEncoding(d_model, dropout_rate, max_len)
        self.embbedding = NormalizationEmbeddingLayer()

    def get_tgt_mask(self, size: int) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))

        return mask

    def forward(self, src, tgt):
        src = self.embbedding(src)
        tgt = self.embbedding(tgt)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        out = self.transformer(src, tgt)

#
# class TransformerModel(nn.Module):
#     """Container module with an encoder, a recurrent or transformer module, and a decoder."""
#
#     def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
#         super(TransformerModel, self).__init__()
#         try:
#             from torch.nn import TransformerEncoder, TransformerEncoderLayer
#         except:
#             raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
#         self.model_type = 'Transformer'
#         self.src_mask = None
#         self.pos_encoder = PositionalEncoding(ninp, dropout)
#         encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         self.ninp = ninp
#         self.decoder = nn.Linear(ninp, ntoken)
#
#         self.init_weights()
#
#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask
#
#     def init_weights(self):
#         initrange = 0.1
#         nn.init.zeros_(self.decoder.bias)
#         nn.init.uniform_(self.decoder.weight, -initrange, initrange)
#
#     def forward(self, src, has_mask=True):
#         if has_mask:
#             device = src.device
#             if self.src_mask is None or self.src_mask.size(0) != len(src):
#                 mask = self._generate_square_subsequent_mask(len(src)).to(device)
#                 self.src_mask = mask
#         else:
#             self.src_mask = None
#
#         src = src * math.sqrt(self.ninp)
#         # print(src.size())
#         src = self.pos_encoder(src)
#         # print(src.size())
#         output = self.transformer_encoder(src, self.src_mask)
#         # print(output.size())
#         output = self.decoder(output)
#         # print(output.size())
#         # return F.log_softmax(output, dim=-1)
#         return F.sigmoid(output)