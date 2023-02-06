import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution

from modules.attention import Transformer
from modules.resblock import ResnetBlock, Downsample, Upsample
from modules.pos_embeds import PosEmbeds
from modules.slot_attention import SlotAttentionBase

from models import TanhBijector, SampleDist

from utils import spatial_broadcast, spatial_flatten

_str_to_activation = {
    'relu': nn.ReLU(),
    'elu' : nn.ELU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}
# h_t is determenistic state, stochastic states are prior \hat{z_t} and posterior z_t. Prior aims at predicting posterior

class OC_RSSM(nn.Module):

    def __init__(self, 
                action_size, 
                stoch_size, 
                deter_size,  
                hidden_size, 
                obs_embed_size, 
                activation,
                num_slots):

        super().__init__()

        self.action_size = action_size
        self.stoch_size  = stoch_size   
        self.deter_size  = deter_size   # GRU hidden units
        self.hidden_size = hidden_size  # intermediate fc_layers hidden units 
        self.embedding_size = obs_embed_size
        self.num_slots = num_slots

        self.act_fn = _str_to_activation[activation]
        # self.rnn = nn.GRUCell(self.deter_size, self.deter_size)
        self.oc_rnn = Transformer(
            self.deter_size, 
            4, 
            self.deter_size // 4,
            1,
            0.1,
            self.deter_size
        )

        self.fc_state_action = nn.Linear(self.stoch_size + self.action_size, self.deter_size)
        self.fc_embed_prior = nn.Linear(self.deter_size, self.hidden_size)
        self.fc_state_prior  = nn.Linear(self.hidden_size, 2*self.stoch_size)
        self.fc_embed_posterior = nn.Linear(self.embedding_size + self.deter_size, self.hidden_size)
        self.fc_state_posterior = nn.Linear(self.hidden_size, 2*self.stoch_size)


    def init_state(self, batch_size, device):

        return dict(
            mean = torch.zeros(batch_size, self.num_slots, self.stoch_size).to(device),
            std  = torch.zeros(batch_size, self.num_slots, self.stoch_size).to(device),
            stoch = torch.zeros(batch_size, self.num_slots, self.stoch_size).to(device),
            deter = torch.zeros(batch_size, self.num_slots, self.deter_size).to(device))

    def get_dist(self, mean, std):

        distribution = distributions.Normal(mean, std)
        distribution = distributions.independent.Independent(distribution, 1)
        return distribution

    def observe_step(self, prev_state, prev_action, obs_embed, nonterm=1.0):

        prior = self.imagine_step(prev_state, prev_action, nonterm)
        posterior_embed = self.act_fn(self.fc_embed_posterior(torch.cat([obs_embed, prior['deter']], dim=-1)))
        posterior = self.fc_state_posterior(posterior_embed)
        mean, std = torch.chunk(posterior, 2, dim=-1)
        std = F.softplus(std) + 0.1
        sample = mean + torch.randn_like(mean) * std

        posterior = {'mean': mean, 'std': std, 'stoch': sample, 'deter': prior['deter']}
        return prior, posterior

    def imagine_step(self, prev_state, prev_action, nonterm=1.0):

        state_action = self.act_fn(self.fc_state_action(torch.cat([prev_state['stoch']*nonterm, prev_action], dim=-1)))
        deter = self.oc_rnn(context=state_action, x=prev_state['deter']*nonterm)
        prior_embed = self.act_fn(self.fc_embed_prior(deter))
        mean, std = torch.chunk(self.fc_state_prior(prior_embed), 2, dim=-1)
        std = F.softplus(std) + 0.1
        sample = mean + torch.randn_like(mean) * std

        prior = {'mean': mean, 'std': std, 'stoch': sample, 'deter': deter}
        return prior

    def observe_rollout(self, obs_embed, actions, nonterms, prev_state, horizon):

        priors = []
        posteriors = []

        for t in range(horizon):
            prev_action = actions[t]* nonterms[t]
            prior_state, posterior_state = self.observe_step(prev_state, prev_action, obs_embed[t], nonterms[t])
            priors.append(prior_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state

        priors = self.stack_states(priors, dim=0)
        posteriors = self.stack_states(posteriors, dim=0)

        return priors, posteriors

    def imagine_rollout(self, actor, prev_state, horizon):

        rssm_state = prev_state
        next_states = []

        for t in range(horizon):
            action = actor(torch.cat([rssm_state['stoch'], rssm_state['deter']], dim=-1).detach())
            rssm_state = self.imagine_step(rssm_state, action)
            next_states.append(rssm_state)

        next_states = self.stack_states(next_states)
        return next_states

    def stack_states(self, states, dim=0):

        return dict(
            mean = torch.stack([state['mean'] for state in states], dim=dim),
            std  = torch.stack([state['std'] for state in states], dim=dim),
            stoch = torch.stack([state['stoch'] for state in states], dim=dim),
            deter = torch.stack([state['deter'] for state in states], dim=dim))

    def detach_state(self, state):

        return dict(
            mean = state['mean'].detach(),
            std  = state['std'].detach(),
            stoch = state['stoch'].detach(),
            deter = state['deter'].detach())

    def seq_to_batch(self, state):

        return dict(
            mean = torch.reshape(state['mean'], (state['mean'].shape[0]* state['mean'].shape[1], *state['mean'].shape[2:])),
            std = torch.reshape(state['std'], (state['std'].shape[0]* state['std'].shape[1], *state['std'].shape[2:])),
            stoch = torch.reshape(state['stoch'], (state['stoch'].shape[0]* state['stoch'].shape[1], *state['stoch'].shape[2:])),
            deter = torch.reshape(state['deter'], (state['deter'].shape[0]* state['deter'].shape[1], *state['deter'].shape[2:])))


class OC_ConvEncoder(nn.Module):

    def __init__(self, input_shape, num_blocks=2, in_channels=3, hidden_size=32, slot_size=64, num_slots=6, downsample=True, dropout=0.1):
        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.slot_size = slot_size
        self.num_slots = num_slots
        self.downsample = downsample
        self.dropout = dropout

        self.first_layer = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=hidden_size, 
            kernel_size=3, 
            padding=1
        )
        self.resnet = nn.Sequential(*[ResnetBlock(in_channels=hidden_size, out_channels=hidden_size, dropout=dropout) for _ in range(num_blocks)])
        if downsample:
            self.last_conv = Downsample(in_channels=hidden_size)
        else:
            self.last_conv = nn.Identity()

        resolution = (input_shape[-2] // 2, input_shape[-1] // 2) if downsample else (input_shape[-2], input_shape[-1])
        self.enc_emb = PosEmbeds(hidden_size=hidden_size, resolution=resolution)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, slot_size)
        )
        self.slot_attention = SlotAttentionBase(num_slots=num_slots, dim=slot_size, hidden_dim=slot_size*2)

    def forward(self, inputs):
        if len(inputs.shape) > 4:
            prev_slots = None
            res_slots = []
            # reshaped = inputs.reshape(-1, *self.input_shape)
            for step in range(inputs.shape[0]):
                inputs_i = inputs[step]
                embed = self.first_layer(inputs_i)
                embed = self.resnet(embed)
                embed = self.downsample(embed)
                embed = self.enc_emb(embed)

                embed = spatial_flatten(embed)
                embed = self.layer_norm(embed)
                embed = self.mlp(embed)

                slots = self.slot_attention(embed, slots=prev_slots)
                res_slots.append(slots)
                prev_slots = slots.detach()
            slots = torch.stack(prev_slots)
            slots = torch.reshape(slots, slots.shape[0]*slots.shape[1], *slots.shape[2:])
        else:
            embed = self.first_layer(inputs)
            embed = self.resnet(embed)
            embed = self.downsample(embed)
            embed = self.enc_emb(embed)

            embed = spatial_flatten(embed)
            embed = self.layer_norm(embed)
            embed = self.mlp(embed)

            slots = self.slot_attention(embed)
        return slots


class OC_ConvDecoder(nn.Module):

    def __init__(self, stoch_size, deter_size, output_shape, hidden_size=32, out_channels=3, dropout=0.1, num_slots=6):
        self.num_slots = num_slots

        self.output_shape = output_shape
        self.decoder_initial_size = (output_shape[-1] // 2, output_shape[-1] // 2)
        self.dec_emb = PosEmbeds(hidden_size, self.decoder_initial_size)

        self.dense = nn.Linear(stoch_size + deter_size, hidden_size)

        self.resnet1 = nn.Sequential(*[ResnetBlock(in_channels=hidden_size, out_channels=hidden_size, dropout=dropout) for _ in range(2)])
        self.upsample = Upsample()
        self.resnet2 = ResnetBlock(in_channels=hidden_size, out_channels=out_channels + 1, dropout=dropout)

    def forward(self, inputs):
        embeds = self.dense(inputs)
        embeds = spatial_broadcast(embeds, self.decoder_initial_size)
        embeds = self.dec_emb(embeds)

        embeds = self.resnet1(embeds)
        embeds = self.upsample(embeds)
        embeds = self.resnet2(embeds)

        embeds = embeds.reshape(inputs.shape[0], self.num_slots, *embeds.shape[1:])
        recons, masks = torch.split(embeds, self.out_channels, dim=-3)
        masks = F.softmax(masks, dim=-1)
        recons = recons * masks
        result = torch.sum(recons, dim=-4)

        out_dist = distributions.independent.Independent(
            distributions.Normal(result, 1), len(self.output_shape))
        return out_dist, recons


class OC_DenseDecoder(nn.Module):

    def __init__(self, stoch_size, deter_size, output_shape, n_layers, units, activation, dist):

        super().__init__()

        self.input_size = stoch_size + deter_size
        self.output_shape = output_shape
        self.n_layers = n_layers
        self.units = units
        self.act_fn = _str_to_activation[activation]
        self.dist = dist

        self.dense = nn.Linear(self.input_size, self.units)
        self.transf = Transformer(self.units, n_heads=4, d_head =self.units//4, depth=1, dropout=0.25)

        layers=[]

        for i in range(self.n_layers - 1):
            in_ch = self.units
            out_ch = self.units
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(self.act_fn) 

        layers.append(nn.Linear(self.units, int(np.prod(self.output_shape))))

        self.model = nn.Sequential(*layers)

    def forward(self, features):
        features = self.dense(features)
        features = self.transf(features)
        features = torch.mean(features, dim=1)

        out = self.model(features)

        if self.dist == 'normal':
            return distributions.independent.Independent(
                distributions.Normal(out, 1), len(self.output_shape))
        if self.dist == 'binary':
            return distributions.independent.Independent(
                distributions.Bernoulli(logits =out), len(self.output_shape))
        if self.dist == 'none':
            return out

        raise NotImplementedError(self.dist)


class OC_ActionDecoder(nn.Module):

    def __init__(self, action_size, stoch_size, deter_size, n_layers, units, 
                        activation, min_std=1e-4, init_std=5, mean_scale=5):

        super().__init__()

        self.action_size = action_size
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.units = units  
        self.act_fn = _str_to_activation[activation]
        self.n_layers = n_layers

        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

        self.dense = nn.Linear(self.stoch_size + self.deter_size, self.units)
        self.transf = Transformer(self.units, n_heads=4, d_head =self.units//4, depth=1, dropout=0.25)

        layers = []
        for i in range(self.n_layers - 1):
            in_ch = self.units
            out_ch = self.units
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(self.act_fn)

        layers.append(nn.Linear(self.units, 2*self.action_size))
        self.action_model = nn.Sequential(*layers)

    def forward(self, features, deter=False):
        features = self.dense(features)
        features = self.transf(features)
        features = torch.mean(features, dim=1)

        out = self.action_model(features)
        mean, std = torch.chunk(out, 2, dim=-1) 

        raw_init_std = np.log(np.exp(self._init_std)-1)
        action_mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
        action_std = F.softplus(std + raw_init_std) + self._min_std

        dist = distributions.Normal(action_mean, action_std)
        dist = TransformedDistribution(dist, TanhBijector())
        dist = distributions.independent.Independent(dist, 1)
        dist = SampleDist(dist)

        if deter:
            return dist.mode()
        else:
            return dist.rsample()

    def add_exploration(self, action, action_noise=0.3):

        return torch.clamp(distributions.Normal(action, action_noise).rsample(), -1, 1)
