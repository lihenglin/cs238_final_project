import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoTokenizer

from rlprompt.models import (LMAdaptorModelConfig, SinglePromptModelConfig,
                             make_lm_adaptor_model, make_single_prompt_model)
from rlprompt.modules import SQLModuleConfig, make_sql_module
from rlprompt.trainers import TrainerConfig, make_trainer
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)

from fsc_helpers import (PromptedClassificationRewardConfig,
                         FewShotClassificationDatasetConfig,
                         make_prompted_classification_reward,
                         make_few_shot_classification_dataset)


# Compose default config
config_list = [PromptedClassificationRewardConfig,
                FewShotClassificationDatasetConfig, LMAdaptorModelConfig,
                SinglePromptModelConfig, SQLModuleConfig, TrainerConfig]
cs = compose_hydra_config_store('base_fsc', config_list)


@hydra.main(version_base=None, config_path="./", config_name="fsc_config_sql")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    output_dir = get_hydra_output_dir()

    (train_dataset, val_dataset, test_dataset,
     num_classes, verbalizers, template) = \
        make_few_shot_classification_dataset(config)

    policy_model = make_lm_adaptor_model(config)
    prompt_model = make_single_prompt_model(policy_model, config)
    reward = make_prompted_classification_reward(num_classes, verbalizers, 
                                                 template, config)
    algo_module = make_sql_module(prompt_model, reward, config)
    # algo_module.load_state_dict(torch.load('outputs/2023-03-01/03-58-20/outputs/ckpt/ckpt.step.12000.pth')['model_state_dict'])
    # algo_module.load_state_dict(torch.load('outputs/2023-03-01/18-08-12/outputs/ckpt/ckpt.step.12000.pth')['model_state_dict'])
    algo_module.load_state_dict(torch.load('outputs/2023-03-02/09-56-19/outputs/ckpt/ckpt.step.12000.pth')['model_state_dict'])
    algo_module.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.policy_lm, pad_token='<|endoftext|>')
    output = algo_module._model.generate(source_texts=[], do_sample=True, top_k=config.top_k, top_p=config.top_p, num_beams=config.num_beams)
    generated_prompts = []
    for i in range(len(output['sample_tokens'])):
        generated_prompts.append(tokenizer.convert_tokens_to_string(output['sample_tokens'][i]))
    print(generated_prompts)

if __name__ == "__main__":
    main()
