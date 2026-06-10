import os
from verl import DataProto
import torch
from verl.utils.reward_score.my_data import my_gsm8k as my_gsm8k_boxed
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _select_rm_score_fn(data_source):
    if data_source == 'zoey/gsm8k':
        return my_gsm8k_boxed.compute_score
    raise NotImplementedError(f"Only 'zoey/gsm8k' is supported in my_ppo. Got: {data_source}")


class RewardManager():
    """The reward manager.
    """
    def __init__(self, tokenizer, num_examine, output_dir, adv_estimator=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.output_dir = output_dir
        self.adv_estimator = adv_estimator

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if self.adv_estimator in {"ada_lang_grpo", "ada_lang_grpo_full"}:
                raise ValueError(
                    f"{self.adv_estimator} requires decoded text outputs "
                    "(sequences_strs/response_strs), but rm_scores path only provides tensor rewards."
                )
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        sequences_strs = []
        response_strs = []
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = int(data_item.batch['attention_mask'][:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            sequences_strs.append(sequences_str)
            response_str = self.tokenizer.decode(valid_response_ids)
            response_strs.append(response_str)

            reward_meta = data_item.non_tensor_batch.get('reward_model', None) or {}
            ground_truth = reward_meta.get('ground_truth', data_item.non_tensor_batch.get('ground_truth', None))

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            if data_source != 'zoey/gsm8k':
                raise ValueError(
                    f"my_ppo is configured for zoey/gsm8k only, but got data_source={data_source!r}."
                )
            compute_score_fn = _select_rm_score_fn(data_source)

            # Language format - "reward_model": {"ground_truth": "42", "lang": "fr"}
            target_lang = reward_meta.get("lang", data_item.non_tensor_batch.get("lang", ""))

            output_file = f"{self.output_dir}/{data_source}.jsonl"

            score = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                output_file=output_file,
            )
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor, sequences_strs, response_strs


import ray
import hydra
import multiprocessing

@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        print("System CPU count:", multiprocessing.cpu_count())
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
        print(ray.available_resources())

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id


    output_dir = os.path.join("reward_logs", config.trainer.experiment_name)
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)

    reward_fn = RewardManager(tokenizer=tokenizer,
                              num_examine=0,
                              output_dir=output_dir,
                              adv_estimator=config.algorithm.adv_estimator)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer,
                                  num_examine=1,
                                  output_dir=output_dir,
                                  adv_estimator=config.algorithm.adv_estimator)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    print("Start training")
    trainer.fit()


if __name__ == '__main__':
    main()
