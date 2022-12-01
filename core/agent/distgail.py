import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import os
import numpy as np

from core.network import Network
from core.optimizer import Optimizer
from core.buffer import ReplayBuffer
#from .base import BaseAgent

import pickle
#from .sac import SAC
from .ppo import PPO

class DISTGAIL(PPO):
    """Soft actor critic (SAC) agent.
    Args:
        state_size (int): dimension of state.
        action_size (int): dimension of action.
        hidden_size (int): dimension of hidden unit.
        actor (str): key of actor network class in _network_dict.txt.
        critic (str): key of critic network class in _network_dict.txt.
        head (str): key of head in _head_dict.txt.
        optim_config (dict): dictionary of the optimizer info.
        use_dynamic_alpha (bool): parameter that determine whether to use autotunning entropy adjustment.
        gamma (float): discount factor.
        tau (float): the soft update coefficient (for soft target update).
        buffer_size (int): the size of the memory buffer.
        batch_size (int): the number of samples in the one batch.
        start_train_step (int): steps to start learning.
        static_log_alpha (float): static value used as log alpha when use_dynamic_alpha is false.
        target_update_period (int): period to update the target network (for hard target update) (unit: step)
        run_step (int): the number of total steps.
        lr_decay: lr_decay option which apply decayed weight on parameters of network.
        device (str): device to use.
            (e.g. 'cpu' or 'gpu'. None can also be used, and in this case, the cpu is used.)
    """

    def __init__(
        self,
        discrim="discrim_network",
        head="mlp",
        optim_config={
            "name": "adam",
            "discrim": "adam",
            "discrim_lr": 5e-4
        },
        hidden_size=512,
        buffer_size=50000,
        **kwargs,
    ):

        super(DISTGAIL, self).__init__(**kwargs)

        self.discrim = Network(
            discrim, self.state_size, self.action_size, D_hidden=hidden_size, head=head
        ).to(self.device)

        self.discrim_optimizer = Optimizer(
            optim_config["discrim"], self.discrim.parameters(), lr=optim_config["discrim_lr"]
        )

        self.memory = ReplayBuffer(buffer_size)
        self.memory_expert = ReplayBuffer(buffer_size)
        with open("./expert_data/sac.hopper.continuous.pickle", "rb") as fr:
            data_expert = pickle.load(fr)
        for i in range(len(data_expert)):
            self.memory_expert.store([data_expert[i]])


    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)

        if self.action_type == "continuous":
            mu, std, _ = self.network(self.as_tensor(state))
            z = torch.normal(mu, std) if training else mu
            action = torch.tanh(z)
        else:
            pi, _ = self.network(self.as_tensor(state))
            action = (
                torch.multinomial(pi, 1)
                if training
                else torch.argmax(pi, dim=-1, keepdim=True)
            )
        action = action.cpu().numpy()
        return {"action": action}


    def learn(self):
        transitions = self.memory.sample(self.batch_size)
        for key in transitions.keys():
            transitions[key] = self.as_tensor(transitions[key])

        state = transitions["state"]
        action = transitions["action"]
        #reward = transitions["reward"]  # be substituted by -log(D(s,a))
        next_state = transitions["next_state"]
        done = transitions["done"]

        ################## expert data
        transitions_expert = self.memory_expert.sample(self.batch_size)
        for key in transitions_expert.keys():
            transitions_expert[key] = self.as_tensor(transitions_expert[key])

        state_expert = transitions_expert["state"]
        action_expert = transitions_expert["action"]
        #reward_expert = transitions_expert["reward"]
        #next_state_expert = transitions_expert["next_state"]
        #done_expert = transitions_expert["done"]
        ################### expert data #e


        #####################  discrim learning
        if self.action_type == "continuous":
            gen_output = self.discrim(state, action)
            expert_output = self.discrim(state_expert, action_expert)

            self.discrim_optimizer.zero_grad(set_to_none=True)
            discrim_loss_fn = torch.nn.BCELoss()
            discrim_loss = discrim_loss_fn(gen_output, torch.ones((state.shape[0], 1)).to(self.device)) + \
                           discrim_loss_fn(expert_output, torch.zeros((state.shape[0], 1)).to(self.device))
            discrim_loss.backward()
            self.discrim_optimizer.step()

        else:
            gen_output = self.discrim(state).gather(1, action.long())
            expert_output = self.discrim(state_expert).gather(1, action_expert.long())

            self.discrim_optimizer.zero_grad(set_to_none=True)
            discrim_loss_fn = torch.nn.BCELoss()
            discrim_loss = discrim_loss_fn(gen_output, torch.ones((state.shape[0], 1)).to(self.device)) + \
                discrim_loss_fn(expert_output, torch.zeros((state.shape[0], 1)).to(self.device))
            discrim_loss.backward()
            self.discrim_optimizer.step()
        #####################  discrim learning #e

        #####################  generator learning
        reward = -torch.log(gen_output)
        #####################  generator learning #e

        # set prob_a_old and advantage
        with torch.no_grad():
            if self.action_type == "continuous":
                mu, std, value = self.network(state)
                m = Normal(mu, std)
                z = torch.atanh(torch.clamp(action, -1 + 1e-7, 1 - 1e-7))
                log_prob = m.log_prob(z)
            else:
                pi, value = self.network(state)
                log_prob = pi.gather(1, action.long()).log()
            log_prob_old = log_prob

            next_value = self.network(next_state)[-1]
            delta = reward + (1 - done) * self.gamma * next_value - value
            adv = delta.clone()
            adv, done = adv.view(-1, self.n_step), done.view(-1, self.n_step)
            for t in reversed(range(self.n_step - 1)):
                adv[:, t] += (
                        (1 - done[:, t]) * self.gamma * self._lambda * adv[:, t + 1]
                )

            ret = adv.view(-1, 1) + value

            if self.use_standardization:
                adv = (adv - adv.mean(dim=1, keepdim=True)) / (
                        adv.std(dim=1, keepdim=True) + 1e-7
                )

            adv = adv.view(-1, 1)

        mean_ret = ret.mean().item()

        # start train iteration
        actor_losses, critic_losses, entropy_losses, ratios, probs = [], [], [], [], []
        idxs = np.arange(len(reward))
        for _ in range(self.n_epoch):
            np.random.shuffle(idxs)
            for offset in range(0, len(reward), self.batch_size):
                idx = idxs[offset: offset + self.batch_size]

                _state, _action, _value, _ret, _adv, _log_prob_old = map(
                    lambda x: [_x[idx] for _x in x] if isinstance(x, list) else x[idx],
                    [state, action, value, ret, adv, log_prob_old],
                )

                if self.action_type == "continuous":
                    mu, std, value_pred = self.network(_state)
                    m = Normal(mu, std)
                    z = torch.atanh(torch.clamp(_action, -1 + 1e-7, 1 - 1e-7))
                    log_prob = m.log_prob(z)
                else:
                    pi, value_pred = self.network(_state)
                    m = Categorical(pi)
                    log_prob = m.log_prob(_action.squeeze(-1)).unsqueeze(-1)

                ratio = (log_prob - _log_prob_old).sum(1, keepdim=True).exp()
                surr1 = ratio * _adv
                surr2 = (
                        torch.clamp(
                            ratio, min=1 - self.epsilon_clip, max=1 + self.epsilon_clip
                        )
                        * _adv
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = _value + torch.clamp(
                    value_pred - _value, -self.epsilon_clip, self.epsilon_clip
                )

                critic_loss1 = F.mse_loss(value_pred, _ret)
                critic_loss2 = F.mse_loss(value_pred_clipped, _ret)

                critic_loss = torch.max(critic_loss1, critic_loss2).mean()

                entropy_loss = -m.entropy().mean()

                loss = (
                        actor_loss
                        + self.vf_coef * critic_loss
                        + self.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.clip_grad_norm
                )
                self.optimizer.step()

                probs.append(log_prob.exp().min().item())
                ratios.append(ratio.max().item())
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())

        result = {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy_loss": np.mean(entropy_losses),
            "discrim_loss": discrim_loss.item(),
            "max_ratio": max(ratios),
            "min_prob": min(probs),
            "mean_ret": mean_ret,
        }
        return result

    def process(self, transitions, step):
        result = {}
        # Process per step
        self.memory.store(transitions)
        delta_t = step - self.time_t
        self.time_t = step
        self.learn_stamp += delta_t

        # Process per epi
        if self.learn_stamp >= self.n_step:
            result = self.learn()
            if self.lr_decay:
                self.learning_rate_decay(step)
            self.learn_stamp = 0

        return result

    def save(self, path):
        print(f"...Save model to {path}...")
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                #############
                "discrim": self.discrim.state_dict(),
                ############# e
            },
            os.path.join(path, "ckpt"),
        )

    def load(self, path):
        print(f"...Load model from {path}...")
        checkpoint = torch.load(os.path.join(path, "ckpt"), map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        ################
        self.discrim.load_state_dict(checkpoint["discrim"])
        ################ e
