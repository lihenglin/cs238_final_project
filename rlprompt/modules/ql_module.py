import torch
import copy
from typing import Optional, List, Dict, Any, Union, Tuple

from rlprompt.models import BaseModel
from rlprompt.modules import BaseModule
from rlprompt.rewards import BaseReward
from rlprompt.modules.module_utils import ForwardMode, get_reward_shaping_func
from rlprompt.losses import ql_loss_with_sparse_rewards
from rlprompt.utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QLModule(BaseModule):
    def __init__(
        self,
        model: BaseModel,
        target_model: Optional[BaseModel],
        reward: Optional[BaseReward],
        ql_loss_impl: str,
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

        self._model = model
        if target_model is None:
            self._target_model = copy.deepcopy(self._model)
        else:
            self._target_model = target_model
        # for p1, p2 in zip(self._model.parameters(), self._target_model.parameters()):
        #     if p1.data.ne(p2.data).sum() > 0:
        #         print(False)
        #     print(True) 
        self._reward = reward

        # TODO: could have a sampling procedure that accepts temperature
        self._ql_loss_impl = ql_loss_impl
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
            self._target_model.load_state_dict(self._model.state_dict())

        # Target network update
        # Note that we are assuming `model.parameters()`
        # would yield the same parameter orders.
        # https://towardsdatascience.com/double-deep-q-networks-905dd8325412
        if self._target_update_method == "polyak":
            for param_, param in zip(self._target_model.parameters(),
                                     self._model.parameters()):
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
        if mode != "QL_ON":
            # TODO: Enable training modes other than on-policy
            raise NotImplementedError('Only on-policy sampling is supported now')

        (logits, logits_, output_tokens, output_ids, sequence_lengths) = \
            self._decode_sampling(batch=batch)

        raw_rewards, rewards_log = \
            self.compute_rewards(batch=batch, 
                                  output_tokens=output_tokens,
                                  mode="train")
        shaped_rewards = self._reward_shaping_func(raw_rewards)

        ql_loss, ql_loss_log = ql_loss_with_sparse_rewards(
            implementation=self._ql_loss_impl,
            logits=logits, # (B, L, N)
            logits_=logits_, # (B, L, N)
            actions=output_ids, # (B, L)
            sampled_actions=None,
            rewards=shaped_rewards, # (B, 1)
            sequence_length=sequence_lengths)

        utils.add_prefix_to_dict_keys_inplace(
            rewards_log, prefix=f"{mode}/rewards/")
        utils.add_prefix_to_dict_keys_inplace(
            ql_loss_log, prefix=f"{mode}/")
        ql_loss_log = utils.unionize_dicts([
            rewards_log,
            ql_loss_log,
            {
                f"{mode}/rewards/raw": raw_rewards.mean(),
                f"{mode}/rewards/shaped": shaped_rewards.mean(),
            },
        ])

        return ql_loss, ql_loss_log

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

    def infer(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, Union[torch.Tensor, torch.LongTensor, List[List[str]]]]:
        return self._model.generate(**batch,
                                    do_sample=False,
                                    top_k=self._top_k,
                                    top_p=self._top_p,
                                    num_beams=self._num_beams,
                                    infer=True)

    def _decode_sampling(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]],
               torch.LongTensor, torch.LongTensor]:
        outputs = self._model.generate(**batch,
                                       do_sample=True,
                                       top_k=self._top_k,
                                       top_p=self._top_p,
                                       num_beams=self._num_beams)

        batch_ = {k: v for k, v in batch.items()}
        batch_.update(outputs)

        outputs_ = self._target_model.teacher_forcing(**batch_)

        return (outputs['sample_logits'].contiguous(),
                outputs_['sample_logits'].contiguous(),
                outputs['sample_tokens'],
                outputs['sample_ids'].contiguous(),
                outputs['sample_lengths'].contiguous())