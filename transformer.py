import torch
import torch.nn as nn

from helpers import Vocabulary


class AttentionBlock(nn.Module):
    def __init__(self, inp_dim, att_dim, masked):
        super().__init__()
        self.masked = masked
        self.Wq = nn.Parameter(torch.randn(inp_dim, att_dim))
        self.Wk = nn.Parameter(torch.randn(inp_dim, att_dim))
        self.Wv = nn.Parameter(torch.randn(inp_dim, att_dim))

    def forward(self, x, decoder_output=None):
        queries = torch.matmul(x, self.Wq)
        keys = torch.matmul(x, self.Wk)
        if decoder_output is not None:
            values = torch.matmul(decoder_output, self.Wv)
        else:
            values = torch.matmul(x, self.Wv)

        softmax = nn.Softmax(dim=2)

        if self.masked:
            inf_tens = torch.zeros(*[keys.shape[1]] * 2) + float('-inf')
            mask = torch.triu(inf_tens, diagonal=1)
            scores = (torch.matmul(queries, keys.permute(0, 2, 1)) + mask) / queries.shape[2] ** 0.5
        else:
            scores = torch.matmul(queries, keys.permute(0, 2, 1)) / queries.shape[2] ** 0.5

        softmax_scores = softmax(scores)

        res = torch.matmul(softmax_scores, values)

        return res


class MultiHeadAttention(nn.Module):
    def __init__(self, inp_dim, att_dim, masked, h):
        super().__init__()
        self.heads = nn.ModuleList([AttentionBlock(inp_dim, att_dim, masked) for _ in range(h)])
        self.linear = nn.Linear(inp_dim, inp_dim)

    def forward(self, x):
        outputs = None
        if isinstance(x, (tuple, list)) and len(x) == 2:
            x, outputs = x
        results_list = []
        for attention_block in self.heads:
            res = attention_block(x, outputs)
            results_list.append(res)
        res = torch.cat(results_list, dim=2)
        res = self.linear(res)
        return res


class FeedForwardNetwork(nn.Module):
    def __init__(self, inp_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inp_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, inp_dim, att_dim, h):
        super().__init__()
        self.multh_attention = MultiHeadAttention(inp_dim, att_dim, masked=False, h=h)
        self.layer_norm_first = nn.LayerNorm(inp_dim)

        self.feed_forward = FeedForwardNetwork(inp_dim, inp_dim)
        self.layer_norm_second = nn.LayerNorm(inp_dim)

    def forward(self, x):
        x_transformed = self.multh_attention(x)
        x_transformed = self.layer_norm_first(x + x_transformed)

        fw_res = self.feed_forward(x_transformed)
        fw_res = self.layer_norm_second(x_transformed + fw_res)
        return fw_res


class DecoderBlock(nn.Module):
    def __init__(self, inp_dim, att_dim, h):
        super().__init__()
        self.multh_attention_masked = MultiHeadAttention(inp_dim, att_dim, masked=True, h=h)
        self.layer_norm_first = nn.LayerNorm(inp_dim)

        self.multh_attention = MultiHeadAttention(inp_dim, att_dim, masked=False, h=h)
        self.layer_norm_second = nn.LayerNorm(inp_dim)

        self.feed_forward = FeedForwardNetwork(inp_dim, inp_dim)
        self.layer_norm_third = nn.LayerNorm(inp_dim)

    def forward(self, x):
        encoder_z, target_seq = x
        y_transformed = self.multh_attention_masked(target_seq)
        y_transformed = self.layer_norm_first(target_seq + y_transformed)

        z_transformed = self.multh_attention([encoder_z, y_transformed])
        z_transformed = self.layer_norm_second(y_transformed + z_transformed)

        fw_res = self.feed_forward(z_transformed)
        fw_res = self.layer_norm_third(z_transformed + fw_res)
        return fw_res


class Encoder(nn.Module):
    def __init__(self, N, inp_dim, att_dim, h):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderBlock(inp_dim, att_dim, h) for _ in range(N)])

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, N, inp_dim, att_dim, num_classes, h):
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(inp_dim, att_dim, h) for _ in range(N)])
        self.linear = nn.Linear(inp_dim, num_classes)

    def forward(self, x):
        for block in self.blocks:
            encoded_z = block(x)
        fw_res = self.linear(encoded_z)
        softmax = nn.Softmax(dim=1)
        return softmax(fw_res)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, sentence_len=1000):
        super().__init__()
        self.emb_dim = emb_dim
        self.sentence_len = sentence_len

    def forward(self, x):
        positions = torch.arange(0., self.sentence_len, 1.)

        power = torch.arange(0., self.emb_dim, 2.) / self.emb_dim

        sin_pe = torch.sin(positions / (10000 ** power)[:, None])
        cos_pe = torch.cos(positions / (10000 ** power)[:, None])

        pe = torch.zeros(self.sentence_len, self.emb_dim)

        sin_pe = sin_pe.transpose(1, 0)
        cos_pe = cos_pe.transpose(1, 0)

        pe[:, 0::2] += sin_pe
        pe[:, 1::2] += cos_pe

        return x + pe.unsqueeze(0)


class Transformer(nn.Module):
    def __init__(self, emb_size, vocab, max_len=5):
        super().__init__()
        self.emb_size = emb_size
        self.vocab = vocab
        self.max_len = max_len
        vocab_len = len(vocab)
        self.embeddings = nn.Embedding(vocab_len, emb_size, padding_idx=0)
        self.pos_encoding = PositionalEncoding(emb_size, max_len)
        self.encoder = Encoder(6, emb_size, 64, 8)
        self.decoder = Decoder(6, emb_size, 64, num_classes=vocab_len, h=8)

    def get_vocab_batch_ids(self, sentences):
        batch_idx = []
        for sentence in sentences:
            idx = [self.vocab[w] for w in sentence.split()]
            batch_idx.append(idx)
        for i in range(len(batch_idx)):
            ids = batch_idx[i]
            diff = self.max_len - len(ids)
            if diff > 0:
                batch_idx[i] = ids + [0] * diff
        return batch_idx

    def forward(self, input_seq, target_seq):
        input_seq = self.get_vocab_batch_ids(input_seq)
        target_seq = self.get_vocab_batch_ids(target_seq)

        inp_emb = self.embeddings(torch.tensor(input_seq)) * self.emb_size ** 0.5
        target_emb = self.embeddings(torch.tensor(target_seq)) * self.emb_size ** 0.5

        inp_emb_pos = self.pos_encoding(inp_emb)
        target_emb_pos = self.pos_encoding(target_emb)

        encoder_z = self.encoder(inp_emb_pos)
        res = self.decoder([encoder_z, target_emb_pos])
        return res


if __name__ == '__main__':
    vocab = Vocabulary()
    vocab.load('vocab.json')
    print('vocab len: ', len(vocab))  # vocab len:  313167

    input_seq = ['Hello, how are you doing?', 'I love French toast!']
    output_seq = ['Hi, pretty nice.', 'Great, I will cook it today!']

    transformer = Transformer(512, vocab)
    res = transformer(input_seq, output_seq)
    print(res.shape)  # 2, 5, 313167
