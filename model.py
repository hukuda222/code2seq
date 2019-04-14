import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import pickle
import numpy as np


class Code2Vec(nn.Module):
    def __init__(self, args, terminal_vocab_size,
                 path_element_vocab_size,
                 target_dict, device,
                 path_embed_size=128, terminal_embed_size=128,
                 path_rnn_size=128 * 2, target_embed_size=128,
                 decode_size=320):

        super(Code2Vec, self).__init__()
        self.decode_size = decode_size
        self.encode_size = decode_size
        self.terminal_embed_size = terminal_embed_size
        self.path_embed_size = path_embed_size
        self.path_rnn_size = path_rnn_size
        self.path_rnn_drop = args.path_rnn_drop
        self.embed_drop = args.embed_drop
        self.target_dict = target_dict
        self.device = device
        self.generate_target_size = args.target_length
        self.target_vocab_size = len(target_dict)

        self.terminal_element_embedding = nn.Embedding(
            terminal_vocab_size, terminal_embed_size)
        self.path_element_embedding = nn.Embedding(
            path_element_vocab_size, path_embed_size)
        self.target_element_embedding = nn.Embedding(
            self.target_vocab_size, target_embed_size)

        # 埋め込み層の初期化
        self.terminal_element_embedding.weight =\
            nn.Parameter(torch.rand(terminal_vocab_size, terminal_embed_size) *
                         math.sqrt(1 / terminal_embed_size))
        self.path_element_embedding.weight =\
            nn.Parameter(torch.rand(path_element_vocab_size, path_embed_size) *
                         math.sqrt(1 / path_embed_size))
        self.target_element_embedding.weight =\
            nn.Parameter(torch.rand(self.target_vocab_size,
                                    target_embed_size) *
                         math.sqrt(1 / target_embed_size))

        # pathをrnnでembedingするやつ、双方向なので隠れ層は1/2
        self.path_rnn = nn.LSTM(
            path_embed_size, path_rnn_size // 2,
            num_layers=1, batch_first=True, dropout=self.path_rnn_drop,
            bidirectional=True)

        # encoderの最後の出力をする
        self.input_linear = nn.Linear(
            terminal_embed_size * 2 + path_rnn_size, self.encode_size,
            bias=False)
        self.input_layer_norm = nn.LayerNorm(decode_size)
        self.input_dropout = nn.Dropout(p=self.embed_drop)

        # decoderで使うrnnのcell
        self.decoder_rnn = nn.LSTMCell(target_embed_size, decode_size)
        self.Wa = nn.Linear(decode_size, decode_size)
        self.Whc = nn.Linear(self.encode_size + decode_size, decode_size)
        self.output_linear = nn.Linear(decode_size, self.target_vocab_size)

        self.loss = nn.NLLLoss(reduction="none")

    def forward(self, starts, paths, ends, targets, context_mask, start_mask,
                path_length, end_mask, target_mask, is_eval):

        batch, max_e, terminal_subword_size = starts.size()
        # embed,terminalはfastText,pathのみbilstm
        embed_element_start = self.terminal_element_embedding(
            starts.view(batch, -1)).\
            view(batch, max_e, terminal_subword_size,
                 self.terminal_embed_size)
        # こいつは(batch,max_e,decode_size)のはず
        embed_fasttext_start = torch.sum(
            embed_element_start * start_mask.unsqueeze(-1), 2)

        embed_element_end = self.terminal_element_embedding(
            ends.view(batch, -1)).\
            view(batch, max_e, terminal_subword_size,
                 self.terminal_embed_size)
        # こいつも(batch,max_e,terminal_embed__size)のはず
        embed_fasttext_end = torch.sum(
            embed_element_end * start_mask.unsqueeze(-1), 2)

        _, _, path_size = paths.size()
        embed_element_path = self.path_element_embedding(
            paths.view(batch, -1)).view(batch * max_e, path_size,
                                        self.path_embed_size)

        # paddingを良い感じに処理するやつ

        ordered_len, ordered_idx = path_length.view(
            batch * max_e, ).sort(0, descending=True)
        embed_element_path_packed = nn.utils.rnn.pack_padded_sequence(
            embed_element_path, ordered_len, batch_first=True)

        # 隠れ層の初期値はall_zeroのやつ こいつは(batch*max_e, path_rnn_size)
        _, (hn, cn) = self.path_rnn(
            embed_element_path_packed)

        # sortしたのを元に戻す
        rnn_embed_path = torch.index_select(
            hn.view(batch * max_e, self.path_rnn_size), 0, ordered_idx).\
            view(batch, max_e, self.path_rnn_size)

        # (batch,max_e,path_rnn_size+decode_size*2)
        combined_context_vectors = torch.cat(
            (embed_fasttext_start, rnn_embed_path, embed_fasttext_end), dim=2)

        # FNN, Layer Normalization, tanh
        # ここまでがencoder
        # 最終的にできるのは、(batch,max_e,decode_size)
        combined_context_vectors = self.input_linear(
            combined_context_vectors.view(batch * max_e, -1)).\
            view(batch, max_e, -1)
        ccv_size = combined_context_vectors.size()
        combined_context_vectors = self.input_layer_norm(
            combined_context_vectors.view(-1, self.decode_size)).view(ccv_size)
        combined_context_vectors = torch.tanh(combined_context_vectors)
        combined_context_vectors = self.input_dropout(
            combined_context_vectors)

        # どうせ使わないので<bos>が入っている一つ目のやつを消してる
        if not is_eval:
            outputs = self.train_decode(
                combined_context_vectors, context_mask, targets)
            return torch.sum(self.loss(outputs[:, 1:].permute(0, 2, 1),
                                       targets[:, 1:]) * target_mask) / batch
        else:
            outputs = self.valid_decode(
                combined_context_vectors, context_mask, targets)
            return self.eval(outputs[:, 1:], targets[:, 1:])

    def eval(self, output, targets):
        true_positive, false_positive, false_negative = 0, 0, 0
        batch, _, _ = output.size()
        predict = torch.argmax(output, 2)
        for pre, tar in zip(predict, targets):
            for pre_word in pre:
                if pre_word == self.target_dict["<eos>"]:
                    break
                elif pre_word != self.target_dict["<pad>"] and \
                        pre_word != self.target_dict["<unk>"]:
                    if pre_word in tar:
                        true_positive += 1
                    else:
                        false_positive += 1
            for tar_word in tar:
                if tar_word != self.target_dict["<pad>"] and \
                    tar_word != self.target_dict["<unk>"] and \
                        tar_word != self.target_dict["<eos>"]:
                    if tar_word not in pre:
                        false_negative += 1

        return true_positive, false_positive, false_negative

    def train_decode(self, encode_context, context_mask, targets):
        batch, max_e, _ = encode_context.size()
        true_output = self.target_element_embedding(targets.view(
            batch * self.generate_target_size, -1)).\
            view(batch, self.generate_target_size, -1)

        context_length = torch.sum(context_mask > 0)
        # (batch,encode_size)/(batch,1) のはず
        init_state = torch.sum(encode_context, 1) / context_length

        # encode_contextは (batch,max_e,encode_size)であるが、encode_size=decode_size
        h_t = init_state.clone()
        c_t = init_state.clone()
        all_output = torch.zeros(
            batch, 1, self.target_vocab_size).to(self.device)

        for i in range(self.generate_target_size - 1):
            # h_tは(batch,decode_size)
            h_t, c_t = self.decoder_rnn(
                true_output[:, i], (h_t, c_t))

            # self.Wa(h_t)で、(batch,encode_size,1)
            # encode_context (batch,max_e,encode_size)
            attn = torch.bmm(encode_context, self.Wa(h_t).unsqueeze(-1))
            # attentionのmask部分を0にする
            # attn:(batch,max_e,1)
            # context_mask:(batch,max_e)
            attn = attn.squeeze(-1) * context_mask

            attn_weight = F.softmax(attn, dim=1)
            # context:(batch,1,encode_size) ->(batch,encode_size)
            context = torch.bmm(attn_weight.unsqueeze(1),
                                encode_context).squeeze(1)
            h_tc = torch.cat([h_t, context], dim=1)
            output = F.tanh(self.Whc(h_tc))
            output_ = F.log_softmax(
                self.output_linear(output), dim=1).unsqueeze(1)
            all_output = torch.cat([all_output, output_], dim=1)

        return all_output

    def valid_decode(self, encode_context, context_mask, targets):
        batch, max_e, _ = encode_context.size()
        output = targets[:, 0].clone()
        output = self.target_element_embedding(output)
        """
        self.target_element_embedding(torch.ones(
            batch, 1,  dtype=torch.long).to(self.device)\
              *self.target_dict["<bos>"])
        """
        context_length = torch.sum(context_mask > 0)

        # (batch,encode_size)/(batch,1)
        init_state = torch.sum(encode_context, 1) / context_length

        # encode_contextは (batch,max_e,encode_size)であるが、encode_size=decode_size
        h_t = init_state.clone()
        c_t = init_state.clone()
        all_output = torch.zeros(
            batch, 1, self.target_vocab_size).to(self.device)

        for i in range(self.generate_target_size - 1):
            # h_tは(batch,decode_size)
            h_t, c_t = self.decoder_rnn(output, (h_t, c_t))

            # self.Wa(h_t)で、(batch,encode_size,1)
            # encode_context (batch,max_e,encode_size)
            attn = torch.bmm(encode_context, self.Wa(h_t).unsqueeze(-1))
            # attentionのmask部分を0にする
            # attn:(batch,max_e,1)
            # context_mask:(batch,max_e)
            attn = attn.squeeze(-1) * context_mask

            attn_weight = F.softmax(attn, dim=1)
            # context:(batch,1,encode_size)
            context = torch.bmm(attn_weight.unsqueeze(1),
                                encode_context).squeeze(1)
            h_tc = torch.cat([h_t, context], dim=1)
            output = F.tanh(self.Whc(h_tc))
            # こっちはあとで使う
            output_ = F.log_softmax(
                self.output_linear(output), dim=1)
            output = torch.argmax(output_, dim=1)
            output = self.target_element_embedding(output)

            all_output = torch.cat([all_output, output_.unsqueeze(1)], dim=1)

        return all_output
