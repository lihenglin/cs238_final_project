from dataclasses import dataclass
from typing import Optional

from rlprompt.modules import SQLModule, SACModule, QLModule
from rlprompt.models import BaseModel
from rlprompt.rewards import BaseReward

def make_sql_module(model: BaseModel,
                    reward: BaseReward,
                    config: "DictConfig",
                    target_model: Optional[BaseModel] = None) -> SQLModule:
    return SQLModule(model, target_model, reward, 
                     config.sql_loss_impl, config.training_mode, 
                     config.mix_strategy, config.target_update_method, 
                     config.target_update_steps, config.target_learning_rate, 
                     config.reward_shaping, config.reward_shaping_old_min, 
                     config.reward_shaping_old_max, 
                     config.reward_shaping_new_min, 
                     config.reward_shaping_new_max, 
                     config.top_k, config.top_p, config.num_beams)

def make_sac_module(actor_model: BaseModel,
                    online_critic_model: BaseModel,
                    reward: BaseReward,
                    config: "DictConfig",
                    target_critic_model: Optional[BaseModel] = None) -> SACModule:
    return SACModule(actor_model, online_critic_model, target_critic_model, reward, 
                     config.sac_loss_impl, config.target_entropy_ratio, 
                     config.forward_mode, config.target_update_method, 
                     config.target_update_steps, config.target_learning_rate, 
                     config.reward_shaping, config.reward_shaping_old_min, 
                     config.reward_shaping_old_max, 
                     config.reward_shaping_new_min, 
                     config.reward_shaping_new_max, 
                     config.top_k, config.top_p, config.num_beams)

def make_ql_module(model: BaseModel,
                   reward: BaseReward,
                   config: "DictConfig",
                   target_model: Optional[BaseModel] = None) -> SQLModule:
    return QLModule(model, target_model, reward, 
                    config.ql_loss_impl, config.forward_mode, config.target_update_method, 
                    config.target_update_steps, config.target_learning_rate, 
                    config.reward_shaping, config.reward_shaping_old_min, 
                    config.reward_shaping_old_max, 
                    config.reward_shaping_new_min, 
                    config.reward_shaping_new_max, 
                    config.top_k, config.top_p, config.num_beams)

@dataclass
class SQLModuleConfig:
    sql_loss_impl: str = "v2_v2r_v3_v3r"
    training_mode: str = "sql-onpolicy"
    mix_strategy: Optional[str] = None
    # Target model setting
    target_update_method: str = "polyak"
    target_update_steps: Optional[int] = None
    target_learning_rate: float = 0.001
    # Reward shaping linearly transforms reward range of [old_min, old_max]
    # to [new_min, new_max]
    reward_shaping: bool = True
    reward_shaping_old_min: float = 0
    reward_shaping_old_max: float = 100
    reward_shaping_new_min: float = -10
    reward_shaping_new_max: float = 10
    # Prompt generation setting
    top_k: Optional[int] = None
    top_p: float = 1.0
    num_beams: int = 1

@dataclass
class SACModuleConfig:
    sac_loss_impl: str = "v2_v2r_v3_v3r"
    target_entropy_ratio: float = 0.98
    forward_mode: str = "SAC_ON"
    # Target model setting
    target_update_method: str = "polyak"
    target_update_steps: Optional[int] = None
    target_learning_rate: float = 0.001
    # Reward shaping linearly transforms reward range of [old_min, old_max]
    # to [new_min, new_max]
    reward_shaping: bool = True
    reward_shaping_old_min: float = 0
    reward_shaping_old_max: float = 100
    reward_shaping_new_min: float = -10
    reward_shaping_new_max: float = 10
    # Prompt generation setting
    top_k: Optional[int] = None
    top_p: float = 1.0
    num_beams: int = 1

@dataclass
class QLModuleConfig:
    ql_loss_impl: str = "v1"
    forward_mode: str = "QL_ON"
    # Target model setting
    target_update_method: str = "polyak"
    target_update_steps: Optional[int] = None
    target_learning_rate: float = 0.001
    # Reward shaping linearly transforms reward range of [old_min, old_max]
    # to [new_min, new_max]
    reward_shaping: bool = True
    reward_shaping_old_min: float = 0
    reward_shaping_old_max: float = 100
    reward_shaping_new_min: float = -10
    reward_shaping_new_max: float = 10
    # Prompt generation setting
    top_k: Optional[int] = None
    top_p: float = 1.0
    num_beams: int = 1