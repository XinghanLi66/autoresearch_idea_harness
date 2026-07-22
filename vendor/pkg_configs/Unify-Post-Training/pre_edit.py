"""Package-level fixes for the Unify-Post-Training MLS-Bench integration.

These patches keep the task focused on the HPT controller rather than
infrastructure issues:

1. Add an HF rollout with the same on/off-policy interfaces as the repo's
   vLLM mix rollout so single-GPU evaluation can use rollout.name=hf.
2. Wire that rollout into mix_fsdp_worker.py.
3. Guard the actor's `offline_loss_type="sft"` branch so a pure-GRPO run with
   no off-policy samples does not produce NaNs.
4. Preserve `whether_off` when off-policy batches are merged back so the
   `off_sft` branch can distinguish off-policy RL from off-policy SFT.
5. Make `off_sft` and `grpo_split` robust when a batch contains pure-SFT
   questions with no remaining on-policy samples.
"""

_MIX_HF_ROLLOUT = """\
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask
from verl.workers.rollout.hf_rollout import HFRollout


def _pre_process_inputs_right_pad(pad_token_id, prompt_token_ids: torch.Tensor):
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    if len(non_pad_index) == 0:
        return []
    return prompt_token_ids[: non_pad_index[-1][0] + 1].tolist()


class MIXHFRollout(HFRollout):
    def generate_on_sequences(self, prompts: DataProto, on_num: int = None) -> DataProto:
        on_num = on_num or self.config.n
        if on_num > 1:
            prompts = prompts.repeat(repeat_times=on_num, interleave=True)

        output = super().generate_sequences(prompts)
        response_length = output.batch["responses"].size(1)
        prefix_mask = torch.zeros(
            (output.batch.batch_size[0], response_length),
            dtype=torch.bool,
            device=output.batch["responses"].device,
        )
        output.batch["prefix_mask"] = prefix_mask
        return output

    @torch.no_grad()
    def generate_off_sequences(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        tgt_input_ids = prompts.batch["tgt_input_ids"]

        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        tgt_list = [
            _pre_process_inputs_right_pad(self.module.config.pad_token_id, tgt_input_ids[i])
            for i in range(batch_size)
        ]
        tgt_list = [tgt + [eos_token_id] if len(tgt) > 0 else tgt for tgt in tgt_list]

        response_length = self.config.response_length
        response = torch.full(
            (batch_size, response_length),
            fill_value=self.module.config.pad_token_id,
            dtype=idx.dtype,
            device=idx.device,
        )
        prefix_mask = torch.zeros(
            (batch_size, response_length), dtype=torch.bool, device=idx.device
        )

        for i, tgt in enumerate(tgt_list):
            resp_len = min(len(tgt), response_length)
            if resp_len > 0:
                response[i, :resp_len] = torch.tensor(
                    tgt[:resp_len], device=idx.device, dtype=idx.dtype
                )
                prefix_mask[i, :resp_len] = True

        seq = torch.cat([idx, response], dim=-1)

        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_eos_mask(
            response_id=response,
            eos_token=eos_token_id,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "tgt_input_ids": tgt_input_ids,
                "prefix_mask": prefix_mask,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, meta_info={"prefix_ratios": [1.0] * batch_size})
"""

_SFT_ONLY_GUARD = """\
                        on_policy_mask = ~off_policy_mask
                        if on_policy_mask.any():
                            on_policy_logprob = log_prob[on_policy_mask]
                            on_policy_old_logprob = old_log_prob[on_policy_mask]

                            # The on-policy advantages should not be computed together with the off-policy rewards.
                            on_policy_advantages = advantages[on_policy_mask]
                            on_policy_eos_mask = response_mask[on_policy_mask]

                            pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                                old_log_prob=on_policy_old_logprob, log_prob=on_policy_logprob,
                                advantages=on_policy_advantages,
                                eos_mask=on_policy_eos_mask,
                                cliprange=clip_ratio,
                                loss_remove_token_mean=self.config.loss_remove_token_mean,
                                loss_remove_clip=self.config.loss_remove_clip
                            )
                        else:
                            pg_loss = log_prob.new_tensor(0.0)
                            pg_clipfrac = log_prob.new_tensor(0.0)
                            ppo_kl = log_prob.new_tensor(0.0)
"""

_WHETHER_OFF_MERGE_GUARD = """\
                                # Merge into original batch
                                if 'whether_off' in combined_off_batch.batch and 'whether_off' not in batch.batch:
                                    batch.batch['whether_off'] = torch.tensor([False] * batch.batch.batch_size[0], dtype=torch.bool)
"""

_GRPO_SPLIT_GUARD = """\
def compute_grpo_outcome_advantage_split(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   on_policy_mask: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   use_std: bool = True):
    \"\"\"
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    For hybrid batches, group statistics are estimated from on-policy samples
    when available. If a question has been fully switched to off-policy data,
    the function falls back to zero-mean, unit-std normalization for that uid.
    \"\"\"
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        all_indices = index.tolist() if hasattr(index, "tolist") else list(index)
        unique_indices = list(dict.fromkeys(all_indices))

        for i in range(bsz):
            if on_policy_mask[i].item() is True:
                id2score[index[i]].append(scores[i])

        for idx in unique_indices:
            if len(id2score[idx]) == 0:
                id2mean[idx] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
                id2std[idx] = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
            elif len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
                id2std[idx] = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
            else:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))

        for idx in id2std:
            if id2std[idx].item() == 0:
                id2std[idx] = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)

        for i in range(bsz):
            if use_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = (scores[i] - id2mean[index[i]])
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores
"""

_OFF_SFT_MASK_LOSS_ON_GUARD = """\
    on_policy_token_mask = (~prefix_mask) * eos_mask
    if on_policy_token_mask.sum() > 0:
        if loss_remove_clip is False:
            on_pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, upper_bound)
            on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)
            on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
            on_pg_loss = verl_F.masked_mean(on_pg_losses, on_policy_token_mask)
        else:
            on_pg_loss = verl_F.masked_mean(on_pg_losses, on_policy_token_mask)
            on_pg_clipfrac = torch.tensor(0.0, device=log_prob.device)
    else:
        on_pg_loss = torch.tensor(0.0, device=log_prob.device)
        on_pg_clipfrac = torch.tensor(0.0, device=log_prob.device)
"""

_OFF_SFT_MASK_LOSS_OFF_GUARD = """\
    if off_policy_reshape == "classic_reject_token":
        off_pg_losses = -advantages * reject_coef * off_ratio
    else:
        off_pg_losses = -advantages * off_ratio
    off_rl_token_mask = prefix_mask * eos_mask * off_rl_mask.view(-1, 1)
    if off_rl_token_mask.sum() > 0:
        off_pg_loss = verl_F.masked_mean(off_pg_losses, off_rl_token_mask)
    else:
        off_pg_loss = torch.tensor(0.0, device=log_prob.device)
    off_pg_clipfrac = torch.tensor(0.0, device=log_prob.device)
"""

_FLASH_ATTN_IMPORT_GUARD = """\
try:
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
except ModuleNotFoundError:
    def _missing_flash_attn(*args, **kwargs):
        raise ModuleNotFoundError(
            "flash_attn is required only when use_remove_padding=True. "
            "This MLS-Bench integration runs with use_remove_padding=False."
        )

    pad_input = _missing_flash_attn
    unpad_input = _missing_flash_attn
    rearrange = _missing_flash_attn
    index_first_axis = _missing_flash_attn
"""

_NO_FLASH_ATTN_LINE = "                                                                attn_implementation='eager',\n"

_PAD_BATCH_GUARD = """\
                            for key in batch.batch.keys():
                                if key == 'batch_size':
                                    continue
                                tensor = batch.batch[key]
                                if tensor.dim() == 1:
                                    last_sample = tensor[-1:].repeat(padding_size)
                                else:
                                    repeat_shape = [padding_size] + [1] * (tensor.dim() - 1)
                                    last_sample = tensor[-1:].repeat(*repeat_shape)
                                padded_batch_dict[key] = torch.cat([tensor, last_sample], dim=0)
"""

_PAD_BATCH_WITH_FLAG_GUARD = """\
                        for key in batch.batch.keys():
                            if key == 'batch_size':
                                continue
                            tensor = batch.batch[key]
                            if key == 'whether_pad':
                                last_sample = torch.ones(padding_size, dtype=torch.bool, device=tensor.device)
                            elif tensor.dim() == 1:
                                last_sample = tensor[-1:].repeat(padding_size)
                            else:
                                repeat_shape = [padding_size] + [1] * (tensor.dim() - 1)
                                last_sample = tensor[-1:].repeat(*repeat_shape)
                            padded_batch_dict[key] = torch.cat([tensor, last_sample], dim=0)
"""


OPS = [
    {
        "op": "create",
        "file": "Unify-Post-Training/hpt/verl/verl/mix_src/mix_hf_rollout.py",
        "content": _MIX_HF_ROLLOUT,
    },
    {
        "op": "replace",
        "file": "Unify-Post-Training/hpt/verl/verl/mix_src/mix_trainer.py",
        "start_line": 883,
        "end_line": 885,
        "content": _WHETHER_OFF_MERGE_GUARD,
    },
    {
        "op": "replace",
        "file": "Unify-Post-Training/hpt/verl/verl/mix_src/mix_core_alg.py",
        "start_line": 427,
        "end_line": 434,
        "content": _OFF_SFT_MASK_LOSS_OFF_GUARD,
    },
    {
        "op": "replace",
        "file": "Unify-Post-Training/hpt/verl/verl/mix_src/mix_core_alg.py",
        "start_line": 361,
        "end_line": 368,
        "content": _OFF_SFT_MASK_LOSS_ON_GUARD,
    },
    {
        "op": "replace",
        "file": "Unify-Post-Training/hpt/verl/verl/mix_src/mix_core_alg.py",
        "start_line": 18,
        "end_line": 73,
        "content": _GRPO_SPLIT_GUARD,
    },
    {
        "op": "replace",
        "file": "Unify-Post-Training/hpt/verl/verl/mix_src/mix_fsdp_worker.py",
        "start_line": 261,
        "end_line": 261,
        "content": "            rollout = MIXHFRollout(module=self.actor_module_fsdp, config=self.config.rollout)\n",
    },
    {
        "op": "replace",
        "file": "Unify-Post-Training/hpt/verl/verl/mix_src/mix_fsdp_worker.py",
        "start_line": 259,
        "end_line": 259,
        "content": "            from .mix_hf_rollout import MIXHFRollout\n",
    },
    {
        "op": "replace",
        "file": "Unify-Post-Training/hpt/verl/verl/workers/actor/dp_actor.py",
        "start_line": 34,
        "end_line": 34,
        "content": _FLASH_ATTN_IMPORT_GUARD,
    },
    {
        "op": "replace",
        "file": "Unify-Post-Training/hpt/verl/verl/workers/critic/dp_critic.py",
        "start_line": 34,
        "end_line": 34,
        "content": _FLASH_ATTN_IMPORT_GUARD,
    },
    {
        "op": "replace",
        "file": "Unify-Post-Training/hpt/verl/verl/mix_src/mix_fsdp_worker.py",
        "start_line": 161,
        "end_line": 161,
        "content": _NO_FLASH_ATTN_LINE,
    },
    {
        "op": "replace",
        "file": "Unify-Post-Training/hpt/verl/verl/workers/fsdp_workers.py",
        "start_line": 196,
        "end_line": 196,
        "content": _NO_FLASH_ATTN_LINE,
    },
    {
        "op": "replace",
        "file": "Unify-Post-Training/hpt/verl/verl/workers/fsdp_workers.py",
        "start_line": 635,
        "end_line": 635,
        "content": "                                                                            attn_implementation='eager',\n",
    },
    {
        "op": "replace",
        "file": "Unify-Post-Training/hpt/verl/verl/workers/fsdp_workers.py",
        "start_line": 884,
        "end_line": 884,
        "content": "                                                                            attn_implementation='eager',\n",
    },
    {
        "op": "replace",
        "file": "Unify-Post-Training/hpt/verl/verl/mix_src/mix_trainer.py",
        "start_line": 1065,
        "end_line": 1072,
        "content": _PAD_BATCH_GUARD,
    },
    {
        "op": "replace",
        "file": "Unify-Post-Training/hpt/verl/verl/mix_src/mix_trainer.py",
        "start_line": 1189,
        "end_line": 1195,
        "content": _PAD_BATCH_WITH_FLAG_GUARD,
    }
]
