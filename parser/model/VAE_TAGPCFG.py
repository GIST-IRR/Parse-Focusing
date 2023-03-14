import torch
import torch.nn as nn
import torch.nn.functional as F
from parser.model.PCFG_module import PCFG_module
from parser.modules.res import ResLayer
from torch.nn.modules.activation import MultiheadAttention

from parser.pcfgs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG


class Root_parameterizer(nn.Module):
    def __init__(self, s_dim, z_dim, NT) -> None:
        super().__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.NT = NT
        
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.root_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.NT),
        )

    def forward(self, z):
        b = z.shape[0]
        root_emb = self.root_emb.expand(b, self.s_dim)
        root_prob = self.root_mlp(root_emb).log_softmax(-1)
        return root_prob

class Nonterm_parameterizer(nn.Module):
    def __init__(self, s_dim, z_dim, NT, T) -> None:
        super().__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.NT = NT
        self.T = T
        self.NT_T = self.NT + self.T

        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))

        self.nonterm_mlp = nn.Linear(self.s_dim, (self.NT_T) ** 2)

    def forward(self, z):
        b = z.shape[0]
        nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
            b, self.NT, self.s_dim
        )
        rule_prob = self.nonterm_mlp(nonterm_emb).log_softmax(-1)
        return rule_prob.reshape(b, self.NT, self.NT_T, self.NT_T)

class Term_parameterizer(nn.Module):
    def __init__(self, s_dim, z_dim, T, V) -> None:
        super().__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.T = T
        self.V = V

        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))

        self.term_mlp = nn.Sequential(
            # nn.Linear(self.s_dim+self.z_dim, self.s_dim),
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.V),
        )

    def forward(self, z):
        b = z.shape[0]
        term_emb = self.term_emb.unsqueeze(0).expand(b, -1, -1)

        # z = z.expand(-1, self.T, -1)
        # term_emb = torch.cat([term_emb, z], -1)
        # term_prob = self.term_mlp(term_emb)

        term_prob = self.term_mlp(term_emb).log_softmax(-1)
        # term_prob = self.term_mlp(z).log_softmax(-1)
        return term_prob

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0.1, bidirectional=bidirectional)
        self.o2p = nn.Linear(hidden_size, output_size * 2)

    def forward(self, input):
        embedded = self.embed(input).unsqueeze(1)

        output, hidden = self.gru(embedded, None)
        # mean loses positional info?
        #output = torch.mean(output, 0).squeeze(0) #output[-1] # Take only the last value
        output = output[-1]#.squeeze(0)
        if self.bidirectional:
            output = output[:, :self.hidden_size] + output[: ,self.hidden_size:] # Sum bidirectional outputs
        else:
            output = output[:, :self.hidden_size]

        ps = self.o2p(output)
        mu, logvar = torch.chunk(ps, 2, dim=1)
        z = self.sample(mu, logvar)
        return mu, logvar, z

    def sample(self, mu, logvar):
        eps = torch.randn(mu.size()).to(mu.device)
        std = torch.exp(logvar / 2.0)
        return mu + eps * std

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, word_dropout=1.):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.word_dropout = word_dropout

        self.embed = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size + input_size, hidden_size, n_layers)
        self.z2h = nn.Linear(input_size, hidden_size)
        self.i2h = nn.Linear(hidden_size + input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size + input_size, output_size)
        #self.out = nn.Linear(hidden_size, output_size)

    def sample(self, output, temperature, max_sample=True):
        if max_sample:
            # Sample top value only
            top_i = output.data.topk(1)[1][0][0]

        else:
            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

        input = torch.LongTensor([top_i]).to(output.device)
        return input, top_i

    def forward(self, z, inputs, temperature):
        n_steps = inputs.size(0)
        outputs = torch.zeros(n_steps, 1, self.output_size).to(inputs.device)

        input = torch.LongTensor([SOS_token]).to(inputs.device)

        hidden = self.z2h(z).unsqueeze(0).repeat(self.n_layers, 1, 1)

        for i in range(n_steps):
            output, hidden = self.step(i, z, input, hidden)
            outputs[i] = output

            use_word_dropout = torch.rand() < self.word_dropout
            if use_word_dropout and i < (n_steps - 1):
                unk_input = torch.LongTensor([UNK_token]).to(inputs.device)
                input = unk_input
                continue

            use_teacher_forcing = torch.rand() < temperature
            if use_teacher_forcing:
                input = inputs[i]
            else:
                input, top_i = self.sample(output, temperature)

        return outputs.squeeze(1)

    def generate(self, z, n_steps, temperature):
        outputs = torch.zeros(n_steps, 1, self.output_size).to(z.device)

        input = torch.LongTensor([SOS_token]).to(z.device)

        hidden = self.z2h(z).unsqueeze(0).repeat(self.n_layers, 1, 1)

        for i in range(n_steps):
            output, hidden = self.step(i, z, input, hidden)
            outputs[i] = output
            input, top_i = self.sample(output, temperature)
            #if top_i == EOS: break
        return outputs.squeeze(1)

    def step(self, s, z, input, hidden):
        # print('[DecoderRNN.step] s =', s, 'z =', z.size(), 'i =', input.size(), 'h =', hidden.size())
        input = F.relu(self.embed(input))
        input = torch.cat((input, z), 1)
        input = input.unsqueeze(0)
        output, hidden = self.gru(input, hidden)
        output = output.squeeze(0)
        output = torch.cat((output, z), 1)
        output = self.out(output)
        return output, hidden

class VAE(nn.Module):
    def __init__(self, encoder=None, decoder=None):
        super(VAE, self).__init__()
        if encoder is None:
            self.encoder = EncoderRNN()
        else:
            self.encoder = encoder
        
        if decoder is None:
            self.decoder = DecoderRNN()
        else:
            self.decoder = decoder

        self.steps_seen = 0

    def encode(self, inputs):
        m, l, z = self.encoder(inputs)
        return m, l, z

    def forward(self, inputs, targets, temperature=1.0):
        m, l, z = self.encoder(inputs)
        decoded = self.decoder(z, targets, temperature)
        return m, l, z, decoded

class VAE_TAGPCFG(PCFG_module):
    def __init__(self, args):
        super(VAE_TAGPCFG, self).__init__()
        self.pcfg = PCFG()
        self.part = PartitionFunction()
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.NT_T = self.NT + self.T
        self.V = args.V

        self.s_dim = args.s_dim
        self.z_dim = args.z_dim
        self.w_dim = args.w_dim
        self.h_dim = args.h_dim

        self.root = Root_parameterizer(
            self.s_dim, self.z_dim, self.NT
        )
        self.nonterms = Nonterm_parameterizer(
            self.s_dim, self.z_dim, self.NT, self.T
        )
        self.terms = Term_parameterizer(
            self.s_dim, self.z_dim, self.T, self.V
        )

        self.attn = EncodingLayer(
            self.V, self.w_dim, self.NT, self.T, num_heads=8
        )

        # Partition function
        self.mode = getattr(args, "mode", None)
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def withoutTerm_parameters(self):
        for name, param in self.named_parameters():
            module_name = name.split(".")[0]
            if module_name != "terms":
                yield param

    def rules_similarity(self, rule=None, unary=None):
        if rule is None:
            rule = self.rules["rule"]
        if unary is None:
            unary = self.rules["unary"]

        b = rule.shape[0]
        
        tkl = self.kl_div(unary) # KLD for terminal
        nkl = self.kl_div(rule) # KLD for nonterminal
        tcs = self.cos_sim(unary) # cos sim for terminal
        ncs = self.cos_sim(
            rule.reshape(b, self.NT, -1)
        ) # cos sim for nonterminal
        log_tcs = self.cos_sim(unary, log=True) # log cos sim for terminal
        log_ncs = self.cos_sim(
            rule.reshape(b, self.NT, -1), log=True
        ) # log cos sim for nonterminal
        
        return {
            "kl_term": tkl,
            "kl_nonterm": nkl,
            "cos_term": tcs,
            "cos_nonterm": ncs,
            "log_cos_term": log_tcs,
            "log_cos_nonterm": log_ncs
        }

    @property
    def metrics(self):
        if getattr(self, "_metrics", None) is None:
            self._metrics = self.rules_similarity()
        return self._metrics

    def clear_metrics(self):
        self._metrics = None

    def forward(self, input):
        x = input["word"]
        b, n = x.shape[:2]

        term_z, cs = self.attn(x, self.terms.term_emb)

        root, rule, unary = \
            self.root(term_z), \
            self.nonterms(term_z), \
            self.terms(term_z)

        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            # unary.retain_grad()
            rule.retain_grad()

        return {
            "unary": unary,
            "root": root,
            "rule": rule,
            "unary_attention": term_z,
            "cosine_similarity": cs.mean(-1),
        }

    def loss(self, input, partition=False, max_depth=0, soft=False):
        self.rules = self.forward(input)
        terms = self.term_from_unary(input["word"], self.rules["unary"])

        # # Soft selection
        # label = self.rules['unary_attention'].log()
        # terms = terms + label

        # Hard selection
        label = F.one_hot(
            self.rules['unary_attention'].argmax(-1), num_classes=self.T
        )
        # terms.masked_fill(~label.bool(), -1e9)

        attn = self.rules['unary_attention'].log().masked_fill(~label.bool(), -1e9)
        terms = terms + attn
        # terms = torch.where(~terms.isinf(), terms, -1e9)

        result = self.pcfg(self.rules, terms, lens=input["seq_len"])
        # Partition function
        if partition:
            self.pf = self.part(self.rules, lens=input["seq_len"], mode=self.mode)
            # Renormalization
            if soft:
                return (-result["partition"] + self.rules["kl"]).mean(), self.pf.mean()
            result["partition"] = result["partition"] - self.pf
        # depth-conditioned inside algorithm
        return (-result["partition"]).mean() \
            + self.rules['cosine_similarity'].mean()

    def evaluate(self, input, decode_type, depth=0, depth_mode=False, **kwargs):
        rules = self.forward(input)
        terms = self.term_from_unary(input["word"], rules["unary"])

        if decode_type == "viterbi":
            result = self.pcfg(
                rules,
                terms,
                lens=input["seq_len"],
                viterbi=True,
                mbr=False
            )
        elif decode_type == "mbr":
            result = self.pcfg(
                rules,
                terms,
                lens=input["seq_len"],
                viterbi=False,
                mbr=True
            )
        else:
            raise NotImplementedError

        if depth > 0:
            result["depth"] = self.part(
                rules, depth, mode="length", depth_output="full"
            )
            result["depth"] = result["depth"].exp()

        if "kl" in rules:
            result["partition"] -= rules["kl"]
        return result
