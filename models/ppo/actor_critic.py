import torch
import torch.nn as nn


def build_mlp(in_dim, hidden_sizes, activation=nn.GELU, out_dim=None):
    layers = []
    prev = in_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(prev, size))
        layers.append(activation())
        prev = size
    if out_dim is not None:
        layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def tanh_normal_sample(mu, log_sigma, eps=1e-6):
    sigma = torch.exp(log_sigma)
    normal = torch.distributions.Normal(mu, sigma)
    u = normal.rsample()
    a = torch.tanh(u)
    logp = normal.log_prob(u) - torch.log(1.0 - a.pow(2) + eps)
    logp = logp.sum(dim=-1, keepdim=True)
    return a, logp, u


def tanh_normal_log_prob(mu, log_sigma, u, eps=1e-6):
    sigma = torch.exp(log_sigma)
    normal = torch.distributions.Normal(mu, sigma)
    a = torch.tanh(u)
    logp = normal.log_prob(u) - torch.log(1.0 - a.pow(2) + eps)
    return logp.sum(dim=-1, keepdim=True)


class PPOActor(nn.Module):
    def __init__(
        self,
        encoder,
        encoder_dim,
        action_dim,
        hidden_sizes=(256, 256),
        learned_log_sigma=False,
        log_sigma_bounds=(-20.0, 2.0),
    ):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.trunk = build_mlp(encoder_dim, hidden_sizes)
        last_dim = hidden_sizes[-1] if hidden_sizes else encoder_dim
        self.mu_head = nn.Linear(last_dim, action_dim)
        self.learned_log_sigma = learned_log_sigma
        self.log_sigma_bounds = log_sigma_bounds

        if learned_log_sigma:
            self.log_sigma = nn.Parameter(torch.zeros(action_dim))
            self.log_sigma_head = None
        else:
            self.log_sigma_head = nn.Linear(last_dim, action_dim)
            self.log_sigma = None

    def encode(self, *encoder_inputs):
        with torch.no_grad():
            return self.encoder(*encoder_inputs)

    def forward(self, *encoder_inputs):
        z = self.encode(*encoder_inputs)
        h = self.trunk(z)
        mu = self.mu_head(h)
        if self.learned_log_sigma:
            log_sigma = self.log_sigma.expand_as(mu)
        else:
            log_sigma = self.log_sigma_head(h)
        min_val, max_val = self.log_sigma_bounds
        log_sigma = torch.clamp(log_sigma, min=min_val, max=max_val)
        return mu, log_sigma


class PPOCritic(nn.Module):
    def __init__(
        self,
        encoder,
        encoder_dim,
        hidden_sizes=(256, 256),
    ):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.trunk = build_mlp(encoder_dim, hidden_sizes)
        last_dim = hidden_sizes[-1] if hidden_sizes else encoder_dim
        self.value_head = nn.Linear(last_dim, 1)

    def encode(self, *encoder_inputs):
        with torch.no_grad():
            return self.encoder(*encoder_inputs)

    def forward(self, *encoder_inputs):
        z = self.encode(*encoder_inputs)
        h = self.trunk(z)
        return self.value_head(h)


class PPOActorCritic(nn.Module):
    def __init__(
        self,
        encoder,
        encoder_dim,
        action_dim,
        actor_hidden_sizes=(256, 256),
        critic_hidden_sizes=(256, 256),
        learned_log_sigma=False,
        log_sigma_bounds=(-20.0, 2.0),
    ):
        super().__init__()
        self.actor = PPOActor(
            encoder=encoder,
            encoder_dim=encoder_dim,
            action_dim=action_dim,
            hidden_sizes=actor_hidden_sizes,
            learned_log_sigma=learned_log_sigma,
            log_sigma_bounds=log_sigma_bounds,
        )
        self.critic = PPOCritic(
            encoder=encoder,
            encoder_dim=encoder_dim,
            hidden_sizes=critic_hidden_sizes,
        )

    def act(self, *encoder_inputs):
        mu, log_sigma = self.actor(*encoder_inputs)
        action, logp, _ = tanh_normal_sample(mu, log_sigma)
        value = self.critic(*encoder_inputs)
        return action, logp, value

    def act_deterministic(self, *encoder_inputs):
        mu, _ = self.actor(*encoder_inputs)
        return torch.tanh(mu)
