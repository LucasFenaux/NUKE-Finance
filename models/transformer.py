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

    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first: bool = False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

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
        if self.batch_first:
            x = x + self.pe[:, x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim: int = 5, d_model: int = 64):
        super(EmbeddingLayer, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class StockTransformer(nn.Module):
    def __init__(self, input_dim: int = 5, d_model: int = 64, nhead: int = 4, batch_first: bool = True,
                 dropout_rate: float = 0.1, max_len: int = 5000):
        super(StockTransformer, self).__init__()
        self.nhead = nhead
        # self.transformer = nn.Transformer(d_model, nhead, batch_first=batch_first)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=batch_first)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)
        self.positional_encoding = PositionalEncoding(d_model, dropout_rate, max_len, batch_first=batch_first)
        self.embbedding = EmbeddingLayer(input_dim=input_dim, d_model=d_model)
        # self.downsample1 = nn.Linear(in_features=d_model, out_features=int(math.sqrt(input_dim)))
        # self.downsample2 = nn.Linear(in_features=int(math.sqrt(input_dim)), out_features=input_dim)
        self.downsample = nn.Linear(in_features=d_model, out_features=input_dim)
        self.init_weights()
    #
    # def get_tgt_mask(self, batch_size: int, size: int) -> torch.tensor:
    #     # Generates a squeare matrix where the each row allows one word more to be seen
    #     mask = torch.tril(torch.ones(batch_size*self.nhead, size, size) == 1) # Lower triangular matrix
    #     mask = mask.float()
    #     mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
    #     mask = mask.masked_fill(mask == 1, float(0.0))
    #
    #     return mask

    # def forward(self, src, tgt, tgt_mask=None):
    def forward(self, src):
        src = self.embbedding(src)
        src = self.positional_encoding(src)

        out = self.transformer(src)
        # out = self.downsample2(self.downsample1(out))
        out = self.downsample(out)
        return out

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def get_name():
        return "StockTransformer"


class TradingTransformer(nn.Module):
    def __init__(self, week_model: nn.Module = None, month_model: nn.Module = None, year_model: nn.Module = None,
                 week_dim: int = 5, month_dim: int = 5, year_dim: int = 5, intermediate_dim: int = 20, x_dim: int = 0,
                 sequence_length: int = 100, n_stocks: int = 1000,
                 d_model: int = 64, nhead: int = 4, batch_first: bool = True, dropout_rate: float = 0.1,
                 max_len: int = 5000):
        super(TradingTransformer, self).__init__()
        self.nhead = nhead
        self.n_stocks = n_stocks
        # we use a different encoding layer for each of the incoming predicitions from the other transformers
        self.week_dim = week_dim
        self.x_dim = x_dim
        self.intermediate_dim = intermediate_dim
        if self.week_dim > 0:
            self.week_embedding = EmbeddingLayer(input_dim=week_dim, d_model=d_model)
        self.week_model = week_model

        self.month_dim = month_dim
        if self.month_dim > 0:
            self.month_embedding = EmbeddingLayer(input_dim=month_dim, d_model=d_model)
        self.month_model = month_model

        self.year_dim = year_dim
        if self.year_dim > 0:
            self.year_embedding = EmbeddingLayer(input_dim=year_dim, d_model=d_model)
        self.year_model = year_model

        self.positional_encoding = PositionalEncoding(d_model, dropout_rate, max_len, batch_first=batch_first)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=3*d_model, nhead=nhead, batch_first=batch_first)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)
        self.downsample = nn.Linear(in_features=3*d_model, out_features=intermediate_dim)

        self.flatten = nn.Flatten(start_dim=1)
        self.sequence_length = sequence_length

        self.action_maker = nn.Sequential(nn.Linear(in_features=intermediate_dim*sequence_length + x_dim,
                                                    out_features=1000)
                                          , nn.ReLU(), nn.Linear(in_features=1000, out_features=n_stocks))

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, values, week_input=None, month_input=None, year_input=None):
        # x is the additional information we want to provide to the trading model on top of the other inputs
        # values is the absolute values for the equivalent relative input values at the last given timestep
        # as well as the amount of stocks currently in the portfolio. One can think of it as the current state
        src = []
        week_values, month_values, year_values = values
        if self.week_dim > 0 and week_input is not None:
            week_input = self.week_embedding(week_input)
            # we put positional encoding before
            week_input = self.positional_encoding(week_input)
            week_out = self.week_model(week_input)
            # we take the actual prediction and "teacher force" the actual values that may not have been properly
            # reconstructed
            week_out = torch.cat([week_input[:, :-1, :], week_out[:, -1:, :]], dim=1)
            # and after, as the model is taught to remove it to reproduce the original sequence
            week_out = self.positional_encoding(week_out)
            src.append(torch.cat([week_values, week_out], dim=1))

        elif self.week_dim > 0 and week_input is None:
            # we make a dummy input
            src.append(torch.zeros((x.size(0), self.sequence_length + 1, self.week_dim), device=self.device))

        if self.month_dim > 0 and month_input is not None:
            month_input = self.month_embedding(month_input)
            # we put positional encoding before
            month_input = self.positional_encoding(month_input)
            month_out = self.month_model(month_input)
            # we take the actual prediction and "teacher force" the actual values that may not have been properly
            # reconstructed
            month_out = torch.cat([month_input[:, :-1, :], month_out[:, -1:, :]], dim=1)
            # and after, as the model is taught to remove it to reproduce the original sequence
            month_out = self.positional_encoding(month_out)
            src.append(torch.cat([month_values, month_out], dim=1))
        elif self.month_dim > 0 and month_input is None:
            # we make a dummy input
            src.append(torch.zeros((x.size(0), self.sequence_length + 1, self.month_dim), device=self.device))

        if self.year_dim > 0 and year_input is not None:
            year_input = self.year_embedding(year_input)
            # we put positional encoding before
            year_input = self.positional_encoding(year_input)
            year_out = self.year_model(year_input)
            # we take the actual prediction and "teacher force" the actual values that may not have been properly
            # reconstructed
            year_out = torch.cat([year_input[:, :-1, :], year_out[:, -1:, :]], dim=1)
            # and after, as the model is taught to remove it to reproduce the original sequence
            year_out = self.positional_encoding(year_out)
            src.append(torch.cat([year_values, year_out], dim=1))
        elif self.year_dim > 0 and year_input is None:
            # we make a dummy input
            src.append(torch.zeros((x.size(0), self.sequence_length + 1, self.year_dim), device=self.device))

        # assuming batch_first, sequence second, features third
        src = torch.cat(src, dim=2)

        # we then downsample the dimensions
        out = self.downsample(src)

        # we then incorporate the additional information we have
        if x is not None:
            out = torch.cat([self.flatten(out), x], dim=1)
        else:
            # we don't have any additional information
            out = self.flatten(out)

        out = self.action_maker(out)

        return out

    @staticmethod
    def get_name():
        return "TradingTransformer"


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