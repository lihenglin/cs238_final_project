import torch
import numpy as np
import torch.nn.functional as F
from functools import partial

from typing import Tuple, Dict, Any, Optional

from rlprompt.losses import loss_utils
from rlprompt.utils import utils


def sac_loss_with_sparse_rewards(
        implementation: str,
        logits: torch.Tensor,
        logits_online_critic: torch.Tensor,
        logits_target_critic: torch.Tensor,
        actions: torch.LongTensor,
        sampled_actions: Optional[torch.LongTensor],
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
        log_alpha: torch.Tensor,
        target_entropy: torch.Tensor,
        coefficient: Optional[float] = None,
        margin_constant: Optional[float] = None,
        margin_coefficient: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Soft Actor-Critic Loss Functions with Sparse Rewards

    Arguments:
        implementation: string, which loss function to use
        logits:                        [batch_size, sequence_length, vocab_size]
        logits_online_critic:          [batch_size, sequence_length, vocab_size]
        logits_target_critic:          [batch_size, sequence_length, vocab_size]
        logits_pi:                     [batch_size, sequence_length, vocab_size]
        actions:                       [batch_size, sequence_length]
        rewards:                       [batch_size]
        sequence_length:               [batch_size]
    """
    if implementation not in ["v1", "v2", "v3", "v2_v2r", "v3_v3r", "v2_v2r_v3_v3r"]:
        raise ValueError

    if not torch.is_tensor(rewards):
        raise TypeError

    if rewards.ndim != 1 or logits.shape[0] != rewards.shape[0]:
        raise ValueError

    if implementation == "v1":
        _sac_critic_loss_func = sac_critic_loss_with_sparse_rewards_1

    if implementation == "v2":
        _sac_critic_loss_func = sac_critic_loss_with_sparse_rewards_2

    if implementation == "v3":
        _sac_critic_loss_func = sac_critic_loss_with_sparse_rewards_3

    if implementation == "v2_v2r":
        _sac_critic_loss_func = partial(
            sac_critic_loss_with_sparse_rewards_2_2_reversed,
            coefficient=coefficient)

    if implementation == "v3_v3r":
        _sac_critic_loss_func = partial(
            sac_critic_loss_with_sparse_rewards_3_3_reversed,
            coefficient=coefficient)

    if implementation == "v2_v2r_v3_v3r":
        _sac_critic_loss_func = partial(
            sac_critic_loss_with_sparse_rewards_2_2_reversed_3_3_reversed,
            coefficient=coefficient)

    if logits.shape != logits_online_critic.shape:
        raise ValueError(
            f"`logits.shape` = {logits.shape}, but "
            f"`logits_online_critic.shape` = {logits_online_critic.shape}")

    if logits.shape != logits_target_critic.shape:
        raise ValueError(
            f"`logits.shape` = {logits.shape}, but "
            f"`logits_target_critic.shape` = {logits_target_critic.shape}")

    raw_critic_losses, quantities_to_log = _sac_critic_loss_func(
        logits_online_critic=logits_online_critic,
        logits_target_critic=logits_target_critic,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length
    )
    critic_loss = loss_utils.mask_and_reduce(
        sequence=raw_critic_losses,
        sequence_length=sequence_length
    )
    raw_policy_losses, entropy = sac_policy_loss(
        logits=logits,
        logits_online_critic=logits_online_critic,
        alpha=log_alpha.exp()
    )
    quantities_to_log["entropy"] = entropy
    policy_loss = loss_utils.mask_and_reduce(
        sequence=raw_policy_losses,
        sequence_length=sequence_length
    )
    raw_entropy_losses = sac_entropy_loss(
        log_alpha=log_alpha,
        entropy=entropy,
        target_entropy=target_entropy
    )
    entropy_loss = loss_utils.mask_and_reduce(
        sequence=raw_entropy_losses,
        sequence_length=sequence_length
    )

    loss = critic_loss + policy_loss + entropy_loss
    loss_log = {
        "loss": loss,
        "critic_loss": critic_loss,
        "policy_loss": policy_loss,
        "entropy_loss": entropy_loss,
        "sequence_length": sequence_length.float().mean(),
        "loss-normalized": loss_utils.mask_and_reduce(
            sequence=raw_critic_losses + raw_policy_losses + raw_entropy_losses,
            sequence_length=sequence_length,
            average_across_timesteps=True,
            sum_over_timesteps=False),
    }

    for key, value in quantities_to_log.items():
        masked_mean, masked_min, masked_max = \
            loss_utils.get_masked_mean_min_max(value,
                                               lengths=sequence_length)
        loss_log[f"{key}/min"] = masked_min
        loss_log[f"{key}/max"] = masked_max
        loss_log[f"{key}/mean"] = masked_mean

    return loss, loss_log

def sac_critic_loss_with_sparse_rewards_1(
        logits_online_critic: torch.Tensor,
        logits_target_critic: torch.Tensor,
        actions: torch.LongTensor,
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    Q = loss_utils.gather_2d_on_last_dim(
        tensor=logits_online_critic,
        index=actions,
        shape=actions.shape)
    # use `V` from the target if available
    V_ = logits_target_critic.logsumexp(dim=-1)

    # Build the target `= V_t+1 + r`
    # where we assume the rewards to be sparse
    # i.e., only comes at the final step
    Q_ = torch.zeros_like(Q)
    Q_[:, :-1] = V_[:, 1:]
    Q_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1] = rewards

    raw_losses = F.mse_loss(Q, Q_, reduction="none")
    quantities_to_log = {
        "Q": Q,
        "V": logits_online_critic.logsumexp(dim=-1),
        "Q_": Q_,
        "V_": V_,
    }

    return raw_losses, quantities_to_log


def sac_critic_loss_with_sparse_rewards_2(
        logits_online_critic: torch.Tensor,
        logits_target_critic: torch.Tensor,
        actions: torch.LongTensor,
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
        _recover_mle: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    Q = loss_utils.gather_2d_on_last_dim(
        tensor=logits_online_critic,
        index=actions,
        shape=actions.shape) # (B, L)
    V = logits_online_critic.logsumexp(dim=-1) # (B, L)
    A = Q - V

    # Target outputs
    Q_ = torch.zeros_like(Q)
    A_ = torch.zeros_like(Q)
    V_ = logits_target_critic.logsumexp(dim=-1)
    Q_[:, :-1] = V_[:, 1:]
    A_[:, :-1] = V_[:, 1:] - V_[:, :-1]
    # Terminal V-target is the last V-target before
    # the episode ends, thus depends on `sequence_length`
    terminal_V_ = V_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1]
    Q_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1] = rewards
    A_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1] = rewards - terminal_V_

    if _recover_mle is True:
        utils.colorful_warning("Recover-MLE Mode", bg="red")
        A_ = A.detach() + 1

    raw_losses = F.mse_loss(A, A_, reduction="none")
    quantities_to_log = {
        "Q": Q,
        "V": V,
        "A": A,
        "Q_": Q_,
        "V_": V_,
        "A_": A_
    }

    return raw_losses, quantities_to_log


def sac_critic_loss_with_sparse_rewards_3(
        logits_online_critic: torch.Tensor,
        logits_target_critic: torch.Tensor,
        actions: torch.LongTensor,
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
        freeze_future_steps: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    Q = loss_utils.gather_2d_on_last_dim(
        tensor=logits_online_critic,
        index=actions,
        shape=actions.shape)
    V = logits_online_critic.logsumexp(dim=-1)
    A = Q - V

    # Target outputs
    V_ = logits_target_critic.logsumexp(dim=-1)

    A2 = loss_utils.masked_reverse_cumsum(
        A,
        lengths=sequence_length,
        dim=-1)

    if freeze_future_steps is True:
        # This line of code essentially
        # decompose `A` (with gradient)
        # and cumsum of future `A`
        # (without gradient)
        A2 = (A2 - A).detach() + A

    raw_losses = F.mse_loss(
        A2, rewards.view(-1, 1) - V_,
        reduction="none")

    quantities_to_log = {
        "Q": Q,
        "V": V,
        "A": A,
        "V_": V_,
    }

    return raw_losses, quantities_to_log


def sac_critic_loss_with_sparse_rewards_2_2_reversed(
        logits_online_critic: torch.Tensor,
        logits_target_critic: torch.Tensor,
        actions: torch.LongTensor,
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
        coefficient: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    raw_losses_2, quantities_to_log_2 = sac_critic_loss_with_sparse_rewards_2(
        logits_online_critic=logits_online_critic,
        logits_target_critic=logits_target_critic,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length)

    utils.add_prefix_to_dict_keys_inplace(
        quantities_to_log_2, prefix="0/")

    if coefficient is not None:
        raw_losses_2_r, quantities_to_log_2_r = sac_critic_loss_with_sparse_rewards_2(
            logits_online_critic=logits_target_critic,
            logits_target_critic=logits_online_critic,
            actions=actions,
            rewards=rewards,
            sequence_length=sequence_length)

        raw_losses = (
            coefficient * raw_losses_2 +
            (1 - coefficient) * raw_losses_2_r)

        utils.add_prefix_to_dict_keys_inplace(
            quantities_to_log_2_r, prefix="1/")

        quantities_to_log = utils.unionize_dicts([
            quantities_to_log_2,
            quantities_to_log_2_r,
        ])
    else:
        raw_losses = raw_losses_2
        quantities_to_log = quantities_to_log_2

    return raw_losses, quantities_to_log


def sac_critic_loss_with_sparse_rewards_3_3_reversed(
        logits_online_critic: torch.Tensor,
        logits_target_critic: torch.Tensor,
        actions: torch.LongTensor,
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
        coefficient: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    raw_losses_3, quantities_to_log_3 = sac_critic_loss_with_sparse_rewards_3(
        logits_online_critic=logits_online_critic,
        logits_target_critic=logits_target_critic,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length)

    utils.add_prefix_to_dict_keys_inplace(
        quantities_to_log_3, prefix="0/")

    if coefficient is not None:
        raw_losses_3_r, quantities_to_log_3_r = sac_critic_loss_with_sparse_rewards_3(
            logits_online_critic=logits_target_critic,
            logits_target_critic=logits_online_critic,
            actions=actions,
            rewards=rewards,
            sequence_length=sequence_length)

        raw_losses = (
            coefficient * raw_losses_3 +
            (1 - coefficient) * raw_losses_3_r)

        utils.add_prefix_to_dict_keys_inplace(
            quantities_to_log_3_r, prefix="1/")

        quantities_to_log = utils.unionize_dicts([
            quantities_to_log_3,
            quantities_to_log_3_r,
        ])
    else:
        raw_losses = raw_losses_3
        quantities_to_log = quantities_to_log_3

    return raw_losses, quantities_to_log


def sac_critic_loss_with_sparse_rewards_2_2_reversed_3_3_reversed(
        logits_online_critic: torch.Tensor,
        logits_target_critic: torch.Tensor,
        actions: torch.LongTensor,
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
        coefficient: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    raw_losses_2, quantities_to_log_2 = sac_critic_loss_with_sparse_rewards_2_2_reversed(
        logits_online_critic=logits_online_critic,
        logits_target_critic=logits_target_critic,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length,
        coefficient=coefficient)

    raw_losses_3, quantities_to_log_3 = sac_critic_loss_with_sparse_rewards_3_3_reversed(
        logits_online_critic=logits_online_critic,
        logits_target_critic=logits_target_critic,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length,
        coefficient=coefficient)

    raw_losses = (raw_losses_2 + raw_losses_3) / 2
    # print(rewards)
    # print(raw_losses)

    utils.add_prefix_to_dict_keys_inplace(
        quantities_to_log_2, prefix="v2/")
    utils.add_prefix_to_dict_keys_inplace(
        quantities_to_log_3, prefix="v3/")
    quantities_to_log = utils.unionize_dicts([
        quantities_to_log_2,
        quantities_to_log_3,
    ])
    return raw_losses, quantities_to_log


def sac_policy_loss(
        logits: torch.Tensor,
        logits_online_critic: torch.Tensor,
        alpha: torch.Tensor
) -> torch.Tensor:

    logits_online_critic = logits_online_critic.detach()

    probs = F.softmax(logits, -1) + 1e-8
    entropy = - probs * torch.log(probs)
    entropy = torch.sum(entropy, dim=-1, keepdim=True) # (B, L, 1)
    q = torch.sum(probs * logits_online_critic, dim=-1, keepdim=True)

    entropy = entropy.squeeze(-1) # (B, L)
    q = q.squeeze(-1) # (B, L)

    policy_losses = - q - alpha.detach() * entropy
    return policy_losses, entropy.detach()
    

def sac_entropy_loss(
        log_alpha: torch.Tensor,
        entropy: torch.Tensor,
        target_entropy: torch.Tensor
) -> torch.Tensor:
    assert not entropy.requires_grad
    
    entropy_losses = -log_alpha * (target_entropy - entropy)
    return entropy_losses