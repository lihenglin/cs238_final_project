import os
import hydra
from omegaconf import DictConfig, OmegaConf

from rlprompt.models import (LMAdaptorModelConfig, SinglePromptModelConfig,
                             make_lm_adaptor_model, make_dueling_lm_adaptor_model, make_single_prompt_model)
from rlprompt.modules import SACModuleConfig, make_sac_module
from rlprompt.trainers import TrainerConfig, make_sac_trainer
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)

from fsc_helpers import (PromptedClassificationRewardConfig,
                         FewShotClassificationDatasetConfig,
                         make_prompted_classification_reward,
                         make_few_shot_classification_dataset)


# Compose default config
config_list = [PromptedClassificationRewardConfig,
                FewShotClassificationDatasetConfig, LMAdaptorModelConfig,
                SinglePromptModelConfig, SACModuleConfig, TrainerConfig]
cs = compose_hydra_config_store('base_fsc', config_list)


@hydra.main(version_base=None, config_path="./", config_name="fsc_config_sac")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    output_dir = get_hydra_output_dir()

    (train_dataset, val_dataset, test_dataset,
     num_classes, verbalizers, template) = \
        make_few_shot_classification_dataset(config)
    print('Train Size:', len(train_dataset))
    print('Examples:', train_dataset[:5])
    print('Val Size', len(val_dataset))
    print('Examples:', val_dataset[:5])

    actor_model = make_lm_adaptor_model(config)
    # critic_model = make_lm_adaptor_model(config)
    critic_model = make_dueling_lm_adaptor_model(config)
    actor_prompt_model = make_single_prompt_model(actor_model, config)
    critic_prompt_model = make_single_prompt_model(critic_model, config)
    reward = make_prompted_classification_reward(num_classes, verbalizers, 
                                                 template, config)
    algo_module = make_sac_module(actor_prompt_model, critic_prompt_model, reward, config)

    # Hack for few-shot classification - Each batch contains all examples
    config.train_batch_size = len(train_dataset)
    config.eval_batch_size = len(val_dataset)
    config.save_dir = os.path.join(output_dir, config.save_dir)
    trainer = make_sac_trainer(algo_module, train_dataset, val_dataset, config)
    trainer.train(config=config)


if __name__ == "__main__":
    main()