import torch
import torch.nn as nn
import copy
from typing import Optional, List, Dict, Any, Union, Tuple

from rlprompt.models import BaseModel
from rlprompt.modules import BaseModule
from rlprompt.rewards import BaseReward
from rlprompt.modules.module_utils import ForwardMode, get_reward_shaping_func
from rlprompt.losses import sac_loss_with_sparse_rewards
from rlprompt.utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SACModule(BaseModule):
    def __init__(
        self,
        actor_model: BaseModel,
        online_critic_model: BaseModel,
        target_critic_model: Optional[BaseModel],
        reward: Optional[BaseReward],
        sac_loss_impl: str,
        target_entropy_ratio: float,
        forward_mode: str,
        target_update_method: str,
        target_update_steps: Optional[int],
        target_learning_rate: float,
        reward_shaping: bool,
        reward_shaping_old_min: float,
        reward_shaping_old_max: float,
        reward_shaping_new_min: float,
        reward_shaping_new_max: float,
        top_k: Optional[int],
        top_p: float,
        num_beams: int,
    ):
        super().__init__()
        # Initialize self._model and self._reward
        assert target_update_method in ["copy", "polyak"]
        assert not (top_k is not None and top_p < 1.0), \
               "Only one of top_k or top_p should be selected"
        
        self._actor = actor_model
        self._online_critic = online_critic_model  
        if target_critic_model is None:
            self._target_critic = copy.deepcopy(self._online_critic)
        else:
            self._target_critic = target_critic_model
        self._log_alpha = nn.Parameter(torch.zeros(1, device=next(self._actor.parameters()).device))
        self._target_entropy = -torch.log(torch.tensor(1.0 / 50257)) * target_entropy_ratio # 50257 is the vocabulary sizes of GPT2 family
        self._reward = reward

        self._sac_loss_impl = sac_loss_impl
        self._forward_mode = forward_mode
        self._target_update_method = target_update_method
        self._target_update_steps = target_update_steps
        self._target_learning_rate = target_learning_rate
        self._top_k = top_k
        self._top_p = top_p
        self._num_beams = num_beams

        if reward_shaping is True:
            self._reward_shaping_func = get_reward_shaping_func(
                old_min=reward_shaping_old_min,
                old_max=reward_shaping_old_max,
                new_min=reward_shaping_new_min,
                new_max=reward_shaping_new_max)
        else:
            self._reward_shaping_func = lambda _r: _r

    def _sync_target_model(self) -> None:
        # https://github.com/transedward/pytorch-dqn/blob/master/dqn_learn.py#L221
        if self._target_update_method == "copy":
            self._target_critic.load_state_dict(self._online_critic.state_dict())

        # Target network update
        # Note that we are assuming `model.parameters()`
        # would yield the same parameter orders.
        # https://towardsdatascience.com/double-deep-q-networks-905dd8325412
        if self._target_update_method == "polyak":
            for param_, param in zip(self._target_critic.parameters(),
                                     self._online_critic.parameters()):
                param_.data.copy_((1 - self._target_learning_rate) * param_
                                  + self._target_learning_rate * param)

    def _pre_steps(self, step: int) -> None:
        if self._target_update_method == "polyak":
            self._sync_target_model()
        elif self._target_update_method == "copy" \
                and step % self._target_update_steps == 0:
            self._sync_target_model()

    def forward(self, batch: Dict[str, Any]) -> Tuple[Union[torch.Tensor, Dict],
                                                      Dict[str, Any]]:
        loss_list = []
        loss_log_list = []
        _loss, _loss_log = self._forward(mode=self._forward_mode, batch=batch)
        loss_list.append(_loss)
        loss_log_list.append(_loss_log)

        # https://discuss.pytorch.org/t/get-the-mean-from-a-list-of-tensors/31989/2
        loss = torch.mean(torch.stack(loss_list))
        loss_log = utils.unionize_dicts(loss_log_list)

        return loss, loss_log

    def _forward(
        self,
        mode: str,
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict]:
        if mode != "SAC_ON":
            # TODO: Enable training modes other than on-policy
            raise NotImplementedError('Only on-policy sampling is supported now')

        (logits, logits_online_critic, logits_target_critic, output_tokens, output_ids, sequence_lengths) = \
            self._decode_sampling(batch=batch)

        raw_rewards, rewards_log = \
            self.compute_rewards(batch=batch, 
                                  output_tokens=output_tokens,
                                  mode="train")
        shaped_rewards = self._reward_shaping_func(raw_rewards)

        sac_loss, sac_loss_log = sac_loss_with_sparse_rewards(
            implementation=self._sac_loss_impl,
            logits=logits, # (B, L, N)
            logits_online_critic=logits_online_critic, # (B, L, N)
            logits_target_critic=logits_target_critic, # (B, L, N)
            actions=output_ids, # (B, L)
            sampled_actions=None,
            rewards=shaped_rewards, # (B, 1)
            sequence_length=sequence_lengths,
            log_alpha=self._log_alpha,
            target_entropy=self._target_entropy)

        utils.add_prefix_to_dict_keys_inplace(
            rewards_log, prefix=f"{mode}/rewards/")
        utils.add_prefix_to_dict_keys_inplace(
            sac_loss_log, prefix=f"{mode}/")
        sac_loss_log = utils.unionize_dicts([
            rewards_log,
            sac_loss_log,
            {
                f"{mode}/rewards/raw": raw_rewards.mean(),
                f"{mode}/rewards/shaped": shaped_rewards.mean(),
            },
        ])

        return sac_loss, sac_loss_log

    def compute_rewards(
        self,
        batch: Dict[str, Any],
        output_tokens: List[List[str]],
        to_tensor: bool = True,
        mode: str = "infer"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        rewards_tensor, rewards_log = self._reward(
            **batch,
            output_tokens=output_tokens,
            to_tensor=to_tensor,
            mode=mode)

        rewards_tensor = rewards_tensor.to(device)            
        return rewards_tensor, rewards_log

    # exploit
    def infer(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, Union[torch.Tensor, torch.LongTensor, List[List[str]]]]:
        return self._actor.generate(**batch,
                                    do_sample=False,
                                    top_k=self._top_k,
                                    top_p=self._top_p,
                                    num_beams=self._num_beams,
                                    infer=True)

    # explore
    def _decode_sampling(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]],
               torch.LongTensor, torch.LongTensor]:
        outputs = self._actor.generate(**batch,
                                       do_sample=True,
                                       top_k=self._top_k,
                                       top_p=self._top_p,
                                       num_beams=self._num_beams)

        batch_ = {k: v for k, v in batch.items()}
        batch_.update(outputs)

        outputs_online_critic = self._online_critic.teacher_forcing(**batch_)
        outputs_target_critic = self._target_critic.teacher_forcing(**batch_)

        return (outputs['sample_logits'].contiguous(),
                outputs_online_critic['sample_logits'].contiguous(),
                outputs_target_critic['sample_logits'].contiguous(),
                outputs['sample_tokens'],
                outputs['sample_ids'].contiguous(),
                outputs['sample_lengths'].contiguous())