import torch
import numpy as np
import torch.nn.functional as F
from functools import partial

from typing import Tuple, Dict, Any, Optional

from rlprompt.losses import loss_utils
from rlprompt.utils import utils


def ql_loss_with_sparse_rewards(
        implementation: str,
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        sampled_actions: Optional[torch.LongTensor],
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
        coefficient: Optional[float] = None,
        margin_constant: Optional[float] = None,
        margin_coefficient: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Q Learning Loss Functions with Sparse Rewards

    Arguments:
        implementation: string, which loss function to use
        logits:          [batch_size, sequence_length, vocab_size]
        logits_:         [batch_size, sequence_length, vocab_size]
        logits_pi:       [batch_size, sequence_length, vocab_size]
        actions:         [batch_size, sequence_length]
        rewards:         [batch_size]
        sequence_length: [batch_size]
    """
    if implementation != "v1":
        raise ValueError

    if not torch.is_tensor(rewards):
        raise TypeError

    if rewards.ndim != 1 or logits.shape[0] != rewards.shape[0]:
        raise ValueError

    _ql_loss_func = q_loss_with_sparse_rewards_1

    if logits.shape != logits_.shape:
        raise ValueError(
            f"`logits.shape` = {logits.shape}, but "
            f"`logits_.shape` = {logits_.shape}")

    raw_losses, quantities_to_log = _ql_loss_func(
        logits=logits,
        logits_=logits_,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length
    )

    loss = loss_utils.mask_and_reduce(
        sequence=raw_losses,
        sequence_length=sequence_length)
    loss_log = {
        "loss": loss,
        "sequence_length": sequence_length.float().mean(),
        "loss-normalized": loss_utils.mask_and_reduce(
            sequence=raw_losses,
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


def q_loss_with_sparse_rewards_1(
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    Q = loss_utils.gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    # use `V` from the target if available
    V_ = loss_utils.gather_2d_on_last_dim(
         tensor=logits_,
         index=torch.argmax(logits, dim=-1),
         shape=Q.shape)

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
        "Q_": Q_,
        "V_": V_,
    }

    return raw_losses, quantities_to_log