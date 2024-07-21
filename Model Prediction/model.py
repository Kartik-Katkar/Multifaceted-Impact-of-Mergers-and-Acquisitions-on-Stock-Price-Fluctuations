import torch
import torch.nn as nn
from torch import Tensor

from embedding import *
from config import config

device = config["model"]["device"]


class Transformer(nn.Module):
    # d_model : number of features - embed-dimnesion
    def __init__(
        self,
        input_size: int,
        batch_first: bool,
        d_model: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_heads: int = 8,
        dropout_encoder: float = 0.2,
        dropout_decoder: float = 0.2,
        dim_feedforward_encoder: int = 2048,
        dim_feedforward_decoder: int = 2048,
        num_predicted_features: int = 1,
    ):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads

        # EMBEDDING
        self.enc_embedding = DataEmbedding(input_size, d_model)
        self.dec_embedding = DataEmbedding(input_size, d_model)
        # Encoder

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers, norm=None
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_decoder_layers, norm=None
        )

        self.projection = nn.Linear(
            in_features=d_model, out_features=num_predicted_features
        )

    def generate_square_subsequent_mask(self, dim1: int, dim2: int) -> Tensor:
        return torch.triu(torch.ones(dim1, dim2) * float("-inf"), diagonal=1)

    def forward(self, src: Tensor, tgt: Tensor):
        """
        ARGS: [SOURCE_SEQUENCE_LENGTH, BATCH_SIZE, FEATURE_IN] | [SEQUENCE_LENGTH, FEATURE_IN]  FOR UNBATCHED
               [T,N,E]
                src_mask: the mask for the src sequence to prevent the model from 
                      using data points from the target sequence
                tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence
        OUTPUT: [TARGET_SEQUENCE_LENGTH, BATCH_SIZE, FAETURE_OUT]
        """

        src_mask = self.generate_square_subsequent_mask(tgt.size(0), src.size(0)).to(
            device
        )
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0), tgt.size(0)).to(
            device
        )
        # print(f"src {src.size()}")
        enc_out = self.enc_embedding(src)
        # print(f"enc_out1 {enc_out.size()}")
        enc_out = self.encoder(enc_out)  # MASK
        # print(f"enc_out2 {enc_out.size()}")
        dec_out = self.dec_embedding(tgt)
        # print(f"dec_out {dec_out.size()}")
        dec_out = self.decoder(
            tgt=dec_out, memory=enc_out, tgt_mask=tgt_mask, memory_mask=src_mask
        )
        # print(f"dec_out2 {dec_out.size()}")
        dec_out = self.projection(dec_out)
        return dec_out
