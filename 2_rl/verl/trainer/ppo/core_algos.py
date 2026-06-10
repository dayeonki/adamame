# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict
import math
import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   sequences_strs=None,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        if sequences_strs is not None:
            uid_order = list(dict.fromkeys(index[i] for i in range(bsz)))
            uid2first_idx = {}
            for i in range(bsz):
                if index[i] not in uid2first_idx:
                    uid2first_idx[index[i]] = i
            
            for p_idx, uid in enumerate(uid_order):
                sample = sequences_strs[uid2first_idx[uid]]
                head, tail = sample[:500], sample[-300:]
                body = f'{head}\n[...]\n{tail}' if len(sample) > 800 else sample
                print(f'[grpo] prompt {p_idx} sample response:\n{body}', flush=True)
        
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                # id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
                id2std[idx] = torch.std(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_dr_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                      eos_mask: torch.Tensor,
                                      index: torch.Tensor):
    """
    Dr. GRPO (https://arxiv.org/pdf/2503.20783) advantage estimator.

    Two differences from standard GRPO:
    1. Advantage = R(q, o_i) - mean(R(q, o_1..G))  — no division by group std.
       Removing std avoids inflating gradients for easy/hard questions where all
       rollouts have near-identical rewards (std ≈ 0).
    2. The policy loss uses global token averaging (masked_mean with axis=None),
       not per-response averaging (1/|o_i|). This removes the length bias where
       shorter correct responses get larger per-token gradients than longer ones.
       Note: the existing compute_policy_loss already uses axis=None, so no change
       is needed there.

    Use with: use_kl_loss=False in actor config (no KL penalty).
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2scores = defaultdict(list)
    with torch.no_grad():
        for i in range(scores.shape[0]):
            id2scores[index[i]].append(scores[i])

        id2mean = {idx: torch.mean(torch.stack(v)) for idx, v in id2scores.items()}
        for i in range(scores.shape[0]):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def get_reasoning_format(sequence_str):
    if "Assistant:<COT>" in sequence_str:
        return "COT"
    elif "Assistant:<CODE>" in sequence_str:
        return "CODE"
    elif "Assistant:<ANSWER>" in sequence_str:
        return "DIRECT"
    elif "Assistant:<LONG_COT>" in sequence_str:
        return "LONG_COT"
    else:
        return "UNKNOWN"


def compute_ada_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                               eos_mask: torch.Tensor,
                                               index: torch.Tensor,
                                               sequences_strs: torch.Tensor,
                                               current_training_step: int,
                                               total_training_steps: int,
                                               num_repeat: int,
                                               epsilon: float = 1e-6,
                                               ):
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2format = defaultdict(dict)
    id2mean = {}
    id2std = {}

    def cosine_decay_alpha(rollout_n, cur_format_num, current_training_step, total_training_steps):
        a = rollout_n / cur_format_num
        b = 1
        return b + 0.5 * (a - b) * (1 + math.cos(math.pi * current_training_step / total_training_steps))

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            uid = index[i]
            solution_str = sequences_strs[i]
            _format = get_reasoning_format(solution_str)
            id2format[uid][_format] = id2format[uid].get(_format, 0) + 1

        for key, value in id2format.items():
            rollout_n = sum(value.values())
            assert rollout_n == num_repeat, f"rollout_n: {rollout_n}, num_repeat: {num_repeat}"

        for i in range(bsz):
            uid = index[i]
            _format = get_reasoning_format(sequences_strs[i])
            if _format == "UNKNOWN":
                assert scores[i] == 0., f"score: {scores[i]}"
            cur_format_num = id2format[uid][_format]
            assert cur_format_num > 0, f"cur_format_num: {cur_format_num}"
            alpha = cosine_decay_alpha(rollout_n, cur_format_num, current_training_step, total_training_steps)

            scores[i] = scores[i] * alpha
            id2score[uid].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor,
                                                  gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++. 
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor,
                                    eos_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward 
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, returns


def compute_adamame_outcome_advantage(token_level_rewards: torch.Tensor,
                                      eos_mask: torch.Tensor,
                                      index: torch.Tensor,
                                      sequences_strs,
                                      response_strs,
                                      query_langs,
                                      num_repeat: int,
                                      current_training_step: int,
                                      total_training_steps: int,
                                      query_align_bonus: float = 1.0,
                                      epsilon: float = 1e-6):
    """
    Language-diversity-aware GRPO advantage estimator (adamame).

    Two complementary forces act on the base accuracy reward r_i ∈ {0, 1}:

    1.  Language diversity weight a_i(t):
            a_i(t) = 1 + (n / n_lang_i - 1) * 0.5 * (1 + cos(π·t/T))
        Responses in a rare language within the group get a multiplicative
        boost that is maximal at t=0 and decays to 1.0 at t=T.
        This encourages diverse-language exploration early in training.

    2.  Query-alignment weight q_i(t):
            q_i(t) = 1 + B * max(0.5 * (1 - cos(π·t/T)), 0.1) * [lang_i == query_lang]
        Responses whose language matches the query language get a growing
        bonus (minimum 0.1*B → B) over the course of training.
        This drives convergence toward the query language.

    Final adjusted score: r_i * a_i(t) * q_i(t)
    Then normalised via standard GRPO group mean/std.

    Args:
        token_level_rewards: (bs, response_length)
        eos_mask:            (bs, response_length)
        index:               (bs,) group UIDs (same uid = same prompt)
        sequences_strs:      (bs,) decoded prompt+response strings (for printing samples)
        response_strs:       (bs,) decoded response-only strings (for language detection)
        query_langs:         (bs,) language codes from dataset 'lang' field (e.g. 'ko', 'pt')
        num_repeat:          rollout n (group size)
        current_training_step / total_training_steps: for annealing
        query_align_bonus:   B — maximum query-language alignment bonus
    """
    try:
        from lingua import Language, LanguageDetectorBuilder
        _lingua_detector = LanguageDetectorBuilder.from_languages(
            Language.FRENCH, Language.JAPANESE, Language.KOREAN, Language.THAI, Language.PORTUGUESE,
            Language.ARABIC, Language.ENGLISH, Language.SPANISH, Language.VIETNAMESE, Language.CHINESE,
            Language.BENGALI, Language.SWAHILI, Language.TELUGU, Language.GERMAN,
        ).build()
    except ImportError:
        _lingua_detector = None

    def _detect_response(text, max_chars=500):
        if _lingua_detector is None or not text:
            return 'unknown'
        try:
            lang = _lingua_detector.detect_language_of(text[:max_chars])
            return lang.iso_code_639_1.name.lower() if lang else 'unknown'
        except Exception:
            return 'unknown'

    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1).clone().float()

    # t, T = current_training_step, max(total_training_steps, 1)
    # # diversity_decay: 1 at t=0  →  0 at t=T
    # diversity_decay = 0.5 * (1 + math.cos(math.pi * t / T))
    # # align_grow:     0 at t=0  →  1 at t=T
    # align_grow = 0.5 * (1 - math.cos(math.pi * t / T))

    t, T = current_training_step, max(total_training_steps, 1)
    diversity_decay = 0.5 * (1 + math.cos(math.pi * t / T))
    align_grow = 0.5 * (1 - math.cos(math.pi * t / T))
    align_grow = max(align_grow, 0.1)  # Ensure minimum 10% bonus from start

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        # Detect reasoning trace language from response-only text
        seq_langs = [_detect_response(response_strs[i]) for i in range(bsz)]

        # Query language comes directly from the dataset 'lang' field
        # (already repeated num_repeat times alongside each response)

        # Count language frequency per group
        id2lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            id2lang_count[index[i]][seq_langs[i]] += 1

        # Print per-prompt query vs response language distribution
        uid2resp_lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            uid2resp_lang_count[index[i]][seq_langs[i]] += 1

        uid_order = list(dict.fromkeys(index[i] for i in range(bsz)))  # preserve order
        uid2first_idx = {}
        for i in range(bsz):
            if index[i] not in uid2first_idx:
                uid2first_idx[index[i]] = i

        for p_idx, uid in enumerate(uid_order):
            q_lang = query_langs[uid2first_idx[uid]]
            resp_counts = uid2resp_lang_count[uid]
            n_resp = sum(resp_counts.values())
            resp_str = ', '.join(
                f'{lang}:{cnt}({100*cnt/n_resp:.0f}%)'
                for lang, cnt in sorted(resp_counts.items(), key=lambda x: -x[1])
            )
            print(f'[adamame] prompt {p_idx} query={q_lang} | responses: {resp_str}', flush=True)
            # Print one sample response (truncated) for inspection
            sample = sequences_strs[uid2first_idx[uid]]
            print(f'[adamame] prompt {p_idx} sample response (last 300 chars): ...{sample[-300:]}', flush=True)

        print(f'[adamame] diversity_decay={diversity_decay:.3f} align_grow={align_grow:.3f}', flush=True)

        # Apply per-response multipliers
        for i in range(bsz):
            uid = index[i]
            lang_i = seq_langs[i]
            query_lang = query_langs[i]
            n_lang_i = max(id2lang_count[uid][lang_i], 1)

            # a(t): rarer language in group → larger early boost
            diversity_weight = 1.0 + (num_repeat / n_lang_i - 1.0) * diversity_decay

            # q(t): grows toward bonus B for responses matching query language
            matches_query = float(lang_i == query_lang and lang_i != 'unknown')
            align_weight = 1.0 + query_align_bonus * align_grow * matches_query

            scores[i] = scores[i] * diversity_weight * align_weight
            id2score[uid].append(scores[i])

        # GRPO normalisation within each group
        for idx in id2score:
            grp = id2score[idx]
            if len(grp) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                stacked = torch.stack(grp)
                id2mean[idx] = stacked.mean()
                id2std[idx] = stacked.std()

        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_adamame_qlang_outcome_advantage(token_level_rewards: torch.Tensor,
                                            eos_mask: torch.Tensor,
                                            index: torch.Tensor,
                                            sequences_strs,
                                            response_strs,
                                            query_langs,
                                            num_repeat: int,
                                            current_training_step: int,
                                            total_training_steps: int,
                                            epsilon: float = 1e-6):
    """
    adamame without the query-language alignment component.

    Only the language diversity weight a_i(t) is applied:
        a_i(t) = 1 + (n / n_lang_i - 1) * 0.5 * (1 + cos(π·t/T))
    Responses in a rare language within the group get a multiplicative boost
    that decays from maximum at t=0 to 1.0 at t=T.

    No query-alignment bonus — useful as an ablation of the qlang component.
    """
    try:
        from lingua import Language, LanguageDetectorBuilder
        _lingua_detector = LanguageDetectorBuilder.from_languages(
            Language.FRENCH, Language.JAPANESE, Language.KOREAN, Language.THAI, Language.PORTUGUESE,
            Language.ARABIC, Language.ENGLISH, Language.SPANISH, Language.VIETNAMESE, Language.CHINESE,
            Language.BENGALI, Language.SWAHILI, Language.TELUGU, Language.GERMAN,
        ).build()
    except ImportError:
        _lingua_detector = None

    def _detect_response(text, max_chars=500):
        if _lingua_detector is None or not text:
            return 'unknown'
        try:
            lang = _lingua_detector.detect_language_of(text[:max_chars])
            return lang.iso_code_639_1.name.lower() if lang else 'unknown'
        except Exception:
            return 'unknown'

    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1).clone().float()

    t, T = current_training_step, max(total_training_steps, 1)
    diversity_decay = 0.5 * (1 + math.cos(math.pi * t / T))

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        # Detect reasoning trace language from response-only text
        seq_langs = [_detect_response(response_strs[i]) for i in range(bsz)]

        id2lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            id2lang_count[index[i]][seq_langs[i]] += 1

        uid2resp_lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            uid2resp_lang_count[index[i]][seq_langs[i]] += 1

        uid_order = list(dict.fromkeys(index[i] for i in range(bsz)))
        uid2first_idx = {}
        for i in range(bsz):
            if index[i] not in uid2first_idx:
                uid2first_idx[index[i]] = i

        for p_idx, uid in enumerate(uid_order):
            q_lang = query_langs[uid2first_idx[uid]]
            resp_counts = uid2resp_lang_count[uid]
            n_resp = sum(resp_counts.values())
            resp_str = ', '.join(
                f'{lang}:{cnt}({100*cnt/n_resp:.0f}%)'
                for lang, cnt in sorted(resp_counts.items(), key=lambda x: -x[1])
            )
            print(f'[adamame_qlang] prompt {p_idx} query={q_lang} | responses: {resp_str}', flush=True)

        print(f'[adamame_qlang] diversity_decay={diversity_decay:.3f} (no align component)', flush=True)

        for i in range(bsz):
            uid = index[i]
            lang_i = seq_langs[i]
            n_lang_i = max(id2lang_count[uid][lang_i], 1)

            diversity_weight = 1.0 + (num_repeat / n_lang_i - 1.0) * diversity_decay
            scores[i] = scores[i] * diversity_weight
            id2score[uid].append(scores[i])

        # GRPO normalisation within each group
        for idx in id2score:
            grp = id2score[idx]
            if len(grp) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                stacked = torch.stack(grp)
                id2mean[idx] = stacked.mean()
                id2std[idx] = stacked.std()

        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


_LINGUA_DETECTOR = None

# Persistent state for feedback-controlled variant (lives for the duration of one training run)
_ADAMAME_FEEDBACK_STATE = {
    'eta_ema':          None,   # EMA of mean normalised per-group entropy
    'diversity_factor': 1.0,    # last computed diversity weight factor
    'align_factor':     0.0,    # last computed alignment factor
}

_ADAMAME_LAGRANGIAN_STATE = {
    'lambda': 0.0,   # Lagrange multiplier; grows when match rate < target, shrinks otherwise
    'mhat':   None,  # EMA of observed match rate; None until first batch
}


def _build_lingua_detector():
    try:
        from lingua import Language, LanguageDetectorBuilder
        return LanguageDetectorBuilder.from_languages(
            Language.FRENCH, Language.JAPANESE, Language.KOREAN, Language.THAI, Language.PORTUGUESE,
            Language.ARABIC, Language.ENGLISH, Language.SPANISH, Language.VIETNAMESE, Language.CHINESE,
            Language.BENGALI, Language.SWAHILI, Language.TELUGU, Language.GERMAN,
        ).build()
    except ImportError:
        return None


def _get_lingua_detector():
    global _LINGUA_DETECTOR
    if _LINGUA_DETECTOR is None:
        _LINGUA_DETECTOR = _build_lingua_detector()
    return _LINGUA_DETECTOR


def compute_adamame_batch_metrics(adv_estimator, response_strs, query_langs,
                                   current_training_step, total_training_steps,
                                   query_align_bonus=1.0):
    """Compute per-step schedule values and language alignment rate for logging.

    Returns a flat dict of train/adamame/* metrics ready to merge into the
    trainer's metrics dict.
    """
    t, T = current_training_step, max(total_training_steps, 1)
    sched_align_grow = max(0.5 * (1 - math.cos(math.pi * t / T)), 0.1)
    diversity_decay = 0.5 * (1 + math.cos(math.pi * t / T))

    # Per-variant effective values
    if adv_estimator in ('adamame_const', 'adamame_tworwd', 'adamame_tworwd_correctonly',
                         'adamame_allcorrect'):
        align_grow = 1.0          # no schedule; B acts as a constant multiplier
        diversity_decay = 0.0
    elif adv_estimator == 'adamame_phased':
        PHASE_SWITCH = 0.7
        phase = t / max(T, 1)
        align_grow = 0.0 if phase < PHASE_SWITCH else 1.0
        diversity_decay = 0.0
    elif adv_estimator == 'adamame_qlang':
        align_grow = 0.0          # no alignment component
    elif adv_estimator in ('adamame_align', 'adamame_postnorm'):
        align_grow = sched_align_grow
        diversity_decay = 0.0
    elif adv_estimator == 'adamame_feedback':
        # Feedback variant: read from module state updated inside the function
        align_grow = _ADAMAME_FEEDBACK_STATE['align_factor']
        diversity_decay = _ADAMAME_FEEDBACK_STATE['diversity_factor']
    elif adv_estimator == 'adamame_lagrangian':
        align_grow = _ADAMAME_LAGRANGIAN_STATE['lambda']
        diversity_decay = 0.0
    else:
        align_grow = sched_align_grow

    # Language match rate: fraction of batch responses whose detected lang == query lang
    detector = _get_lingua_detector()
    bsz = len(response_strs)
    n_match = sum(
        1 for i in range(bsz)
        if _detect_lang(detector, response_strs[i]) == query_langs[i] != 'unknown'
    )
    lang_match_rate = n_match / max(bsz, 1)

    out = {
        'train/adamame/align_grow':      align_grow,
        'train/adamame/diversity_decay': diversity_decay,
        'train/adamame/effective_q':     query_align_bonus * align_grow,
        'train/adamame/lang_match_rate': lang_match_rate,
    }

    if adv_estimator == 'adamame_feedback':
        out['train/adamame/eta_ema']          = _ADAMAME_FEEDBACK_STATE['eta_ema'] or 0.0
        out['train/adamame/diversity_factor'] = _ADAMAME_FEEDBACK_STATE['diversity_factor']
        out['train/adamame/align_factor']     = _ADAMAME_FEEDBACK_STATE['align_factor']

    if adv_estimator == 'adamame_lagrangian':
        out['train/adamame/lambda'] = _ADAMAME_LAGRANGIAN_STATE['lambda']
        out['train/adamame/mhat']   = _ADAMAME_LAGRANGIAN_STATE['mhat'] or 0.0

    return out


def _detect_lang(detector, text, max_chars=500):
    if detector is None or not text:
        return 'unknown'
    try:
        lang = detector.detect_language_of(text[:max_chars])
        return lang.iso_code_639_1.name.lower() if lang else 'unknown'
    except Exception:
        return 'unknown'


def _grpo_normalize(scores, index, epsilon=1e-6):
    id2score = defaultdict(list)
    for i in range(scores.shape[0]):
        id2score[index[i]].append(scores[i])
    id2mean, id2std = {}, {}
    for idx, grp in id2score.items():
        if len(grp) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        else:
            stacked = torch.stack(grp)
            id2mean[idx] = stacked.mean()
            id2std[idx] = stacked.std()
    for i in range(scores.shape[0]):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    return scores


def _print_lang_dist(tag, uid_order, uid2first_idx, uid2resp_lang_count, query_langs, extra=''):
    for p_idx, uid in enumerate(uid_order):
        q_lang = query_langs[uid2first_idx[uid]]
        resp_counts = uid2resp_lang_count[uid]
        n_resp = sum(resp_counts.values())
        resp_str = ', '.join(
            f'{lang}:{cnt}({100*cnt/n_resp:.0f}%)'
            for lang, cnt in sorted(resp_counts.items(), key=lambda x: -x[1])
        )
        print(f'[{tag}] prompt {p_idx} query={q_lang} | responses: {resp_str}', flush=True)
    if extra:
        print(f'[{tag}] {extra}', flush=True)


def compute_adamame_align_outcome_advantage(token_level_rewards: torch.Tensor,
                                            eos_mask: torch.Tensor,
                                            index: torch.Tensor,
                                            sequences_strs,
                                            response_strs,
                                            query_langs,
                                            num_repeat: int,
                                            current_training_step: int,
                                            total_training_steps: int,
                                            query_align_bonus: float = 1.0,
                                            epsilon: float = 1e-6):
    """
    adamame with only the query-language alignment component (no diversity weight).

    Only q_i(t) is applied:
        q_i(t) = 1 + B * max(0.5 * (1 - cos(π·t/T)), 0.1) * [lang_i == query_lang]
    Responses matching the query language get a growing bonus over training.

    Ablation that isolates the query-alignment signal without language diversity weighting.
    """
    try:
        from lingua import Language, LanguageDetectorBuilder
        _lingua_detector = LanguageDetectorBuilder.from_languages(
            Language.FRENCH, Language.JAPANESE, Language.KOREAN, Language.THAI, Language.PORTUGUESE,
            Language.ARABIC, Language.ENGLISH, Language.SPANISH, Language.VIETNAMESE, Language.CHINESE,
            Language.BENGALI, Language.SWAHILI, Language.TELUGU, Language.GERMAN,
        ).build()
    except ImportError:
        _lingua_detector = None

    def _detect_response(text, max_chars=500):
        if _lingua_detector is None or not text:
            return 'unknown'
        try:
            lang = _lingua_detector.detect_language_of(text[:max_chars])
            return lang.iso_code_639_1.name.lower() if lang else 'unknown'
        except Exception:
            return 'unknown'

    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1).clone().float()

    t, T = current_training_step, max(total_training_steps, 1)
    align_grow = 0.5 * (1 - math.cos(math.pi * t / T))
    align_grow = max(align_grow, 0.1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        seq_langs = [_detect_response(response_strs[i]) for i in range(bsz)]

        uid2resp_lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            uid2resp_lang_count[index[i]][seq_langs[i]] += 1

        uid_order = list(dict.fromkeys(index[i] for i in range(bsz)))
        uid2first_idx = {}
        for i in range(bsz):
            if index[i] not in uid2first_idx:
                uid2first_idx[index[i]] = i

        for p_idx, uid in enumerate(uid_order):
            q_lang = query_langs[uid2first_idx[uid]]
            resp_counts = uid2resp_lang_count[uid]
            n_resp = sum(resp_counts.values())
            resp_str = ', '.join(
                f'{lang}:{cnt}({100*cnt/n_resp:.0f}%)'
                for lang, cnt in sorted(resp_counts.items(), key=lambda x: -x[1])
            )
            print(f'[adamame_align] prompt {p_idx} query={q_lang} | responses: {resp_str}', flush=True)

        print(f'[adamame_align] align_grow={align_grow:.3f} (no diversity component)', flush=True)

        for i in range(bsz):
            uid = index[i]
            lang_i = seq_langs[i]
            query_lang = query_langs[i]

            matches_query = float(lang_i == query_lang and lang_i != 'unknown')
            align_weight = 1.0 + query_align_bonus * align_grow * matches_query

            scores[i] = scores[i] * align_weight
            id2score[uid].append(scores[i])

        for idx in id2score:
            grp = id2score[idx]
            if len(grp) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                stacked = torch.stack(grp)
                id2mean[idx] = stacked.mean()
                id2std[idx] = stacked.std()

        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_adamame_staged_outcome_advantage(token_level_rewards: torch.Tensor,
                                             eos_mask: torch.Tensor,
                                             index: torch.Tensor,
                                             sequences_strs,
                                             response_strs,
                                             query_langs,
                                             num_repeat: int,
                                             current_training_step: int,
                                             total_training_steps: int,
                                             query_align_bonus: float = 1.0,
                                             epsilon: float = 1e-6):
    """
    Phase-separated adamame (hard staged curriculum).

    Phase 1 (t < T/2): diversity only.
        a_i = 1 + (n / n_lang_i - 1)   [full strength, no decay]
    Phase 2 (t >= T/2): query-language alignment only.
        q_i = 1 + B * [lang_i == query_lang]   [full strength, no ramp]

    No overlap between the two forces: exploration is fully on in phase 1,
    fully off in phase 2; convergence is fully off in phase 1, fully on in phase 2.
    """
    detector = _build_lingua_detector()
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1).clone().float()

    t, T = current_training_step, max(total_training_steps, 1)
    phase = 1 if t < T // 2 else 2

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        seq_langs = [_detect_lang(detector, response_strs[i]) for i in range(bsz)]

        id2lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            id2lang_count[index[i]][seq_langs[i]] += 1

        uid2resp_lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            uid2resp_lang_count[index[i]][seq_langs[i]] += 1
        uid_order = list(dict.fromkeys(index[i] for i in range(bsz)))
        uid2first_idx = {}
        for i in range(bsz):
            if index[i] not in uid2first_idx:
                uid2first_idx[index[i]] = i
        _print_lang_dist('adamame_staged', uid_order, uid2first_idx, uid2resp_lang_count, query_langs,
                         extra=f'phase={phase} (t={t}, T={T})')

        for i in range(bsz):
            uid = index[i]
            lang_i = seq_langs[i]
            query_lang = query_langs[i]
            n_lang_i = max(id2lang_count[uid][lang_i], 1)

            if phase == 1:
                weight = 1.0 + (num_repeat / n_lang_i - 1.0)
            else:
                matches_query = float(lang_i == query_lang and lang_i != 'unknown')
                weight = 1.0 + query_align_bonus * matches_query

            scores[i] = scores[i] * weight
            id2score[uid].append(scores[i])

        for idx in id2score:
            grp = id2score[idx]
            if len(grp) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                stacked = torch.stack(grp)
                id2mean[idx] = stacked.mean()
                id2std[idx] = stacked.std()

        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_adamame_rcalign_outcome_advantage(token_level_rewards: torch.Tensor,
                                              eos_mask: torch.Tensor,
                                              index: torch.Tensor,
                                              sequences_strs,
                                              response_strs,
                                              query_langs,
                                              num_repeat: int,
                                              current_training_step: int,
                                              total_training_steps: int,
                                              query_align_bonus: float = 1.0,
                                              epsilon: float = 1e-6):
    """
    Reward-conditional alignment adamame.

    Two-step process:
    1. Apply diversity weight a_i(t) to raw scores; GRPO-normalise into advantages.
    2. Scale the normalised advantages by q_i(t):
           q_i(t) = 1 + B * align_grow * [lang_i == query_lang]

    Applying alignment post-normalisation means:
      - correct + query-lang  → positive advantage scaled UP   (stronger reward)
      - correct + other-lang  → positive advantage unchanged
      - wrong   + query-lang  → negative advantage scaled DOWN (stronger penalty)
      - wrong   + other-lang  → negative advantage unchanged

    This creates a sharper gradient than pre-normalisation weighting, directly
    linking language choice to the sign of the advantage signal.
    """
    detector = _build_lingua_detector()
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1).clone().float()

    t, T = current_training_step, max(total_training_steps, 1)
    diversity_decay = 0.5 * (1 + math.cos(math.pi * t / T))
    align_grow = max(0.5 * (1 - math.cos(math.pi * t / T)), 0.1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        seq_langs = [_detect_lang(detector, response_strs[i]) for i in range(bsz)]

        id2lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            id2lang_count[index[i]][seq_langs[i]] += 1

        uid2resp_lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            uid2resp_lang_count[index[i]][seq_langs[i]] += 1
        uid_order = list(dict.fromkeys(index[i] for i in range(bsz)))
        uid2first_idx = {}
        for i in range(bsz):
            if index[i] not in uid2first_idx:
                uid2first_idx[index[i]] = i
        _print_lang_dist('adamame_rcalign', uid_order, uid2first_idx, uid2resp_lang_count, query_langs,
                         extra=f'diversity_decay={diversity_decay:.3f} align_grow={align_grow:.3f}')

        # Step 1: apply diversity weight and GRPO-normalise
        for i in range(bsz):
            uid = index[i]
            lang_i = seq_langs[i]
            n_lang_i = max(id2lang_count[uid][lang_i], 1)
            diversity_weight = 1.0 + (num_repeat / n_lang_i - 1.0) * diversity_decay
            scores[i] = scores[i] * diversity_weight
            id2score[uid].append(scores[i])

        for idx in id2score:
            grp = id2score[idx]
            if len(grp) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                stacked = torch.stack(grp)
                id2mean[idx] = stacked.mean()
                id2std[idx] = stacked.std()

        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        # Step 2: scale normalised advantages by alignment weight
        for i in range(bsz):
            lang_i = seq_langs[i]
            query_lang = query_langs[i]
            matches_query = float(lang_i == query_lang and lang_i != 'unknown')
            align_weight = 1.0 + query_align_bonus * align_grow * matches_query
            scores[i] = scores[i] * align_weight

        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_adamame_const_outcome_advantage(token_level_rewards: torch.Tensor,
                                            eos_mask: torch.Tensor,
                                            index: torch.Tensor,
                                            sequences_strs,
                                            response_strs,
                                            query_langs,
                                            num_repeat: int,
                                            current_training_step: int,
                                            total_training_steps: int,
                                            query_align_bonus: float = 1.0,
                                            epsilon: float = 1e-6):
    """
    Constant query-alignment bonus, no schedule (adamame_11).

    q_i = 1 + B * [lang_i == query_lang]

    No cosine ramp — full alignment bonus from step 0.  Tests whether the
    gradual ramp in adamame_3 is helping or simply delaying useful signal.
    """
    detector = _build_lingua_detector()
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1).clone().float()

    with torch.no_grad():
        bsz = scores.shape[0]
        seq_langs = [_detect_lang(detector, response_strs[i]) for i in range(bsz)]

        uid2resp_lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            uid2resp_lang_count[index[i]][seq_langs[i]] += 1
        uid_order = list(dict.fromkeys(index[i] for i in range(bsz)))
        uid2first_idx = {}
        for i in range(bsz):
            if index[i] not in uid2first_idx:
                uid2first_idx[index[i]] = i
        _print_lang_dist('adamame_const', uid_order, uid2first_idx, uid2resp_lang_count, query_langs,
                         extra=f'query_align_bonus={query_align_bonus} (no schedule)')

        for i in range(bsz):
            lang_i = seq_langs[i]
            query_lang = query_langs[i]
            matches_query = float(lang_i == query_lang and lang_i != 'unknown')
            scores[i] = scores[i] * (1.0 + query_align_bonus * matches_query)

        scores = _grpo_normalize(scores, index, epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_adamame_postnorm_outcome_advantage(token_level_rewards: torch.Tensor,
                                               eos_mask: torch.Tensor,
                                               index: torch.Tensor,
                                               sequences_strs,
                                               response_strs,
                                               query_langs,
                                               num_repeat: int,
                                               current_training_step: int,
                                               total_training_steps: int,
                                               query_align_bonus: float = 1.0,
                                               epsilon: float = 1e-6):
    """
    Post-normalization alignment scaling (adamame_13).

    Step 1: GRPO-normalize raw rewards into advantages.
    Step 2: Scale advantages by the alignment weight q_i(t):
                adv_i_final = adv_i * (1 + B * align_grow(t) * [lang_i == query_lang])

    Alignment acts on already-normalized advantages rather than raw rewards,
    so it stretches/compresses the gradient signal without distorting the
    reward scale.  Same q_i(t) schedule as adamame_3 but applied post-norm.
    This isolates whether the issue in adamame_rcalign was the diversity
    pre-step or the post-norm idea itself.
    """
    detector = _build_lingua_detector()
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1).clone().float()

    t, T = current_training_step, max(total_training_steps, 1)
    align_grow = max(0.5 * (1 - math.cos(math.pi * t / T)), 0.1)

    with torch.no_grad():
        bsz = scores.shape[0]
        seq_langs = [_detect_lang(detector, response_strs[i]) for i in range(bsz)]

        uid2resp_lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            uid2resp_lang_count[index[i]][seq_langs[i]] += 1
        uid_order = list(dict.fromkeys(index[i] for i in range(bsz)))
        uid2first_idx = {}
        for i in range(bsz):
            if index[i] not in uid2first_idx:
                uid2first_idx[index[i]] = i
        _print_lang_dist('adamame_postnorm', uid_order, uid2first_idx, uid2resp_lang_count, query_langs,
                         extra=f'align_grow={align_grow:.3f} query_align_bonus={query_align_bonus}')

        # Step 1: standard GRPO normalization on raw rewards
        scores = _grpo_normalize(scores, index, epsilon)

        # Step 2: scale normalized advantages by alignment weight
        for i in range(bsz):
            lang_i = seq_langs[i]
            query_lang = query_langs[i]
            matches_query = float(lang_i == query_lang and lang_i != 'unknown')
            scores[i] = scores[i] * (1.0 + query_align_bonus * align_grow * matches_query)

        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_adamame_tworwd_outcome_advantage(token_level_rewards: torch.Tensor,
                                             eos_mask: torch.Tensor,
                                             index: torch.Tensor,
                                             sequences_strs,
                                             response_strs,
                                             query_langs,
                                             num_repeat: int,
                                             current_training_step: int,
                                             total_training_steps: int,
                                             query_align_bonus: float = 1.0,
                                             epsilon: float = 1e-6):
    """
    Two-reward decomposition (adamame_15).

    Two fully independent GRPO-normalized signals are combined additively:
        adv_i = GRPO_norm(r_i) + B * GRPO_norm([lang_i == query_lang])

    The correctness signal and the language-match signal each get their own
    baseline and variance normalisation, so neither can drown out the other.
    Language signal flows even through wrong responses without distorting the
    correctness baseline.  B (query_align_bonus) controls relative weighting.
    """
    detector = _build_lingua_detector()
    response_length = token_level_rewards.shape[-1]
    scores_correct = token_level_rewards.sum(dim=-1).clone().float()

    with torch.no_grad():
        bsz = scores_correct.shape[0]
        seq_langs = [_detect_lang(detector, response_strs[i]) for i in range(bsz)]

        uid2resp_lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            uid2resp_lang_count[index[i]][seq_langs[i]] += 1
        uid_order = list(dict.fromkeys(index[i] for i in range(bsz)))
        uid2first_idx = {}
        for i in range(bsz):
            if index[i] not in uid2first_idx:
                uid2first_idx[index[i]] = i
        _print_lang_dist('adamame_tworwd', uid_order, uid2first_idx, uid2resp_lang_count, query_langs,
                         extra=f'query_align_bonus={query_align_bonus}')

        # Language-match reward: 1.0 if matches query lang, 0.0 otherwise
        scores_lang = torch.zeros(bsz, dtype=torch.float32)
        for i in range(bsz):
            lang_i = seq_langs[i]
            query_lang = query_langs[i]
            scores_lang[i] = float(lang_i == query_lang and lang_i != 'unknown')

        # Normalize each signal independently, then combine
        scores_correct = _grpo_normalize(scores_correct, index, epsilon)
        scores_lang = _grpo_normalize(scores_lang, index, epsilon)

        scores = scores_correct + query_align_bonus * scores_lang
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_adamame_strat_outcome_advantage(token_level_rewards: torch.Tensor,
                                            eos_mask: torch.Tensor,
                                            index: torch.Tensor,
                                            sequences_strs,
                                            response_strs,
                                            query_langs,
                                            num_repeat: int,
                                            current_training_step: int,
                                            total_training_steps: int,
                                            query_align_bonus: float = 1.0,
                                            epsilon: float = 1e-6):
    """
    Language-stratified advantage normalization (adamame_8).

    Same a_i(t) * q_i(t) weighting as adamame_1, but GRPO normalization is
    computed within (prompt, language) subgroups rather than across the full
    prompt group.  This prevents the majority language from dominating the
    group mean/std, giving rare-language responses a fairer gradient signal.

    Subgroups with only one sample fall back to the full-group statistics so
    the advantage is still comparable across languages within the group.
    """
    detector = _build_lingua_detector()
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1).clone().float()

    t, T = current_training_step, max(total_training_steps, 1)
    diversity_decay = 0.5 * (1 + math.cos(math.pi * t / T))
    align_grow = max(0.5 * (1 - math.cos(math.pi * t / T)), 0.1)

    with torch.no_grad():
        bsz = scores.shape[0]
        seq_langs = [_detect_lang(detector, response_strs[i]) for i in range(bsz)]

        id2lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            id2lang_count[index[i]][seq_langs[i]] += 1

        uid2resp_lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            uid2resp_lang_count[index[i]][seq_langs[i]] += 1
        uid_order = list(dict.fromkeys(index[i] for i in range(bsz)))
        uid2first_idx = {}
        for i in range(bsz):
            if index[i] not in uid2first_idx:
                uid2first_idx[index[i]] = i
        _print_lang_dist('adamame_strat', uid_order, uid2first_idx, uid2resp_lang_count, query_langs,
                         extra=f'diversity_decay={diversity_decay:.3f} align_grow={align_grow:.3f}')

        # Apply same multipliers as adamame_1
        for i in range(bsz):
            uid = index[i]
            lang_i = seq_langs[i]
            query_lang = query_langs[i]
            n_lang_i = max(id2lang_count[uid][lang_i], 1)
            diversity_weight = 1.0 + (num_repeat / n_lang_i - 1.0) * diversity_decay
            matches_query = float(lang_i == query_lang and lang_i != 'unknown')
            align_weight = 1.0 + query_align_bonus * align_grow * matches_query
            scores[i] = scores[i] * diversity_weight * align_weight

        # Full-group stats as fallback for singleton subgroups
        id2scores_all = defaultdict(list)
        for i in range(bsz):
            id2scores_all[index[i]].append(scores[i])
        id2mean_full, id2std_full = {}, {}
        for uid, grp in id2scores_all.items():
            stacked = torch.stack(grp)
            id2mean_full[uid] = stacked.mean()
            id2std_full[uid] = stacked.std() if len(grp) > 1 else torch.tensor(1.0)

        # Stratified normalization within (uid, lang) subgroups
        strat2indices = defaultdict(list)
        for i in range(bsz):
            strat2indices[(index[i], seq_langs[i])].append(i)

        for (uid, lang), idxs in strat2indices.items():
            if len(idxs) == 1:
                i = idxs[0]
                scores[i] = (scores[i] - id2mean_full[uid]) / (id2std_full[uid] + epsilon)
            else:
                grp = torch.stack([scores[j] for j in idxs])
                mean, std = grp.mean(), grp.std()
                for k, i in enumerate(idxs):
                    scores[i] = (grp[k] - mean) / (std + epsilon)

        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_adamame_add_outcome_advantage(token_level_rewards: torch.Tensor,
                                          eos_mask: torch.Tensor,
                                          index: torch.Tensor,
                                          sequences_strs,
                                          response_strs,
                                          query_langs,
                                          num_repeat: int,
                                          current_training_step: int,
                                          total_training_steps: int,
                                          query_align_bonus: float = 1.0,
                                          lambda_diversity: float = 0.5,
                                          lambda_align: float = 0.5,
                                          epsilon: float = 1e-6):
    """
    Additive language bonus decomposition (adamame_9).

    Instead of multiplying a_i(t) * q_i(t) into r_i (which zeroes both
    bonuses when r_i = 0), bonuses are summed independently:

        score_i = r_i
                  + lambda_d * diversity_bonus_i(t)    [decoupled diversity]
                  + lambda_q * align_bonus_i(t)         [decoupled alignment]

    where both bonus terms are normalised to [0, 1]:
        diversity_bonus_i(t) = ((n/n_lang_i - 1) / (n - 1)) * diversity_decay
        align_bonus_i(t)     = query_align_bonus * align_grow * [lang_i == query_lang]

    Wrong responses can still receive language-preference gradient this way.
    Standard GRPO normalisation is applied to the composite score.
    """
    detector = _build_lingua_detector()
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1).clone().float()

    t, T = current_training_step, max(total_training_steps, 1)
    diversity_decay = 0.5 * (1 + math.cos(math.pi * t / T))
    align_grow = max(0.5 * (1 - math.cos(math.pi * t / T)), 0.1)
    norm_denom = max(num_repeat - 1.0, 1.0)

    with torch.no_grad():
        bsz = scores.shape[0]
        seq_langs = [_detect_lang(detector, response_strs[i]) for i in range(bsz)]

        id2lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            id2lang_count[index[i]][seq_langs[i]] += 1

        uid2resp_lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            uid2resp_lang_count[index[i]][seq_langs[i]] += 1
        uid_order = list(dict.fromkeys(index[i] for i in range(bsz)))
        uid2first_idx = {}
        for i in range(bsz):
            if index[i] not in uid2first_idx:
                uid2first_idx[index[i]] = i
        _print_lang_dist('adamame_add', uid_order, uid2first_idx, uid2resp_lang_count, query_langs,
                         extra=f'diversity_decay={diversity_decay:.3f} align_grow={align_grow:.3f} '
                               f'lambda_d={lambda_diversity} lambda_q={lambda_align}')

        for i in range(bsz):
            uid = index[i]
            lang_i = seq_langs[i]
            query_lang = query_langs[i]
            n_lang_i = max(id2lang_count[uid][lang_i], 1)

            diversity_bonus = ((num_repeat / n_lang_i - 1.0) / norm_denom) * diversity_decay
            matches_query = float(lang_i == query_lang and lang_i != 'unknown')
            align_bonus = query_align_bonus * align_grow * matches_query

            scores[i] = scores[i] + lambda_diversity * diversity_bonus + lambda_align * align_bonus

        scores = _grpo_normalize(scores, index, epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_adamame_adaptive_outcome_advantage(token_level_rewards: torch.Tensor,
                                               eos_mask: torch.Tensor,
                                               index: torch.Tensor,
                                               sequences_strs,
                                               response_strs,
                                               query_langs,
                                               num_repeat: int,
                                               current_training_step: int,
                                               total_training_steps: int,
                                               query_align_bonus: float = 1.0,
                                               epsilon: float = 1e-6):
    """
    Distribution-adaptive diversity bonus (adamame_10).

    Replaces the time-based diversity decay with a per-group entropy measure:
        adaptive_factor_g = max(0, 1 - H_g / log(num_repeat))
    where H_g is the Shannon entropy of the language distribution in group g.

    When a group already produces diverse outputs (H_g → log(n)): factor → 0,
    diversity weight → 1 (no extra pressure needed).
    When a group collapses to one language (H_g = 0): factor = 1 (max pressure).

    This is data-driven: diversity pressure fades as soon as the model achieves
    diversity for a given prompt, rather than on a fixed schedule.
    Query-alignment weight q_i(t) remains time-based as in adamame_1.
    """
    detector = _build_lingua_detector()
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1).clone().float()

    t, T = current_training_step, max(total_training_steps, 1)
    align_grow = max(0.5 * (1 - math.cos(math.pi * t / T)), 0.1)

    with torch.no_grad():
        bsz = scores.shape[0]
        seq_langs = [_detect_lang(detector, response_strs[i]) for i in range(bsz)]

        id2lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            id2lang_count[index[i]][seq_langs[i]] += 1

        # Per-group entropy → adaptive diversity factor
        id2adaptive = {}
        H_max = math.log(num_repeat)
        for uid, lang_count in id2lang_count.items():
            n_total = sum(lang_count.values())
            if n_total <= 1:
                id2adaptive[uid] = 1.0
            else:
                probs = [c / n_total for c in lang_count.values()]
                H = -sum(p * math.log(p + 1e-12) for p in probs)
                id2adaptive[uid] = max(0.0, 1.0 - H / H_max)

        uid2resp_lang_count = id2lang_count
        uid_order = list(dict.fromkeys(index[i] for i in range(bsz)))
        uid2first_idx = {}
        for i in range(bsz):
            if index[i] not in uid2first_idx:
                uid2first_idx[index[i]] = i
        adaptive_str = ' '.join(f'{uid}:{id2adaptive[uid]:.2f}' for uid in uid_order[:4])
        _print_lang_dist('adamame_adaptive', uid_order, uid2first_idx, uid2resp_lang_count, query_langs,
                         extra=f'align_grow={align_grow:.3f} | adaptive_factors(first4): {adaptive_str}')

        for i in range(bsz):
            uid = index[i]
            lang_i = seq_langs[i]
            query_lang = query_langs[i]
            n_lang_i = max(id2lang_count[uid][lang_i], 1)

            adaptive_factor = id2adaptive[uid]
            diversity_weight = 1.0 + (num_repeat / n_lang_i - 1.0) * adaptive_factor

            matches_query = float(lang_i == query_lang and lang_i != 'unknown')
            align_weight = 1.0 + query_align_bonus * align_grow * matches_query

            scores[i] = scores[i] * diversity_weight * align_weight

        scores = _grpo_normalize(scores, index, epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_adamame_feedback_outcome_advantage(token_level_rewards: torch.Tensor,
                                               eos_mask: torch.Tensor,
                                               index: torch.Tensor,
                                               sequences_strs,
                                               response_strs,
                                               query_langs,
                                               num_repeat: int,
                                               current_training_step: int,
                                               total_training_steps: int,
                                               query_align_bonus: float = 1.0,
                                               ema_alpha: float = 0.9,
                                               diversity_threshold: float = 0.6,
                                               align_threshold: float = 0.4,
                                               epsilon: float = 1e-6):
    """
    Feedback-controlled adaptive reward (adamame_16).

    Instead of a fixed time schedule, both diversity and alignment pressures
    are driven by an EMA of the observed per-group language entropy η:

        η_g      = H_g / log(num_repeat)   (0 = fully collapsed, 1 = fully diverse)
        η_ema[t] = α * η_ema[t-1] + (1-α) * mean_g(η_g)

    Adaptive factors derived from η_ema:
        diversity_factor = max(0,  1 - η_ema / τ_d)
            Full pressure (1.0) when η_ema=0; linearly fades to 0 at τ_d.
        align_factor     = clamp((η_ema - τ_a) / (1 - τ_a),  0, 1)
            Zero until η_ema reaches τ_a; linearly grows to 1.0 at η_ema=1.

    With τ_a < τ_d (default 0.4 < 0.6) there is an overlap band where both
    pressures coexist, enabling a smooth transition rather than a hard switch.

    Final score: r_i * (1 + (n/n_lang_i - 1) * diversity_factor)
                      * (1 + B * align_factor * [lang_i == query_lang])

    State (η_ema, diversity_factor, align_factor) is kept in a module-level
    dict and persists across steps within one training run.
    """
    detector = _get_lingua_detector()
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1).clone().float()

    H_max = math.log(max(num_repeat, 2))

    with torch.no_grad():
        bsz = scores.shape[0]
        seq_langs = [_detect_lang(detector, response_strs[i]) for i in range(bsz)]

        id2lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            id2lang_count[index[i]][seq_langs[i]] += 1

        # Per-group normalised entropy
        uid2eta = {}
        for uid, lang_count in id2lang_count.items():
            n_total = sum(lang_count.values())
            probs = [c / n_total for c in lang_count.values()]
            H = -sum(p * math.log(p + 1e-12) for p in probs)
            uid2eta[uid] = min(H / H_max, 1.0)

        eta_batch = sum(uid2eta.values()) / max(len(uid2eta), 1)

        # EMA update (initialise to current batch on first call)
        if _ADAMAME_FEEDBACK_STATE['eta_ema'] is None:
            _ADAMAME_FEEDBACK_STATE['eta_ema'] = eta_batch
        else:
            _ADAMAME_FEEDBACK_STATE['eta_ema'] = (
                ema_alpha * _ADAMAME_FEEDBACK_STATE['eta_ema'] + (1.0 - ema_alpha) * eta_batch
            )
        eta_ema = _ADAMAME_FEEDBACK_STATE['eta_ema']

        # Adaptive factors from smoothed entropy
        diversity_factor = max(0.0, 1.0 - eta_ema / max(diversity_threshold, 1e-6))
        denom_align = max(1.0 - align_threshold, 1e-6)
        align_factor = max(0.0, min(1.0, (eta_ema - align_threshold) / denom_align))

        _ADAMAME_FEEDBACK_STATE['diversity_factor'] = diversity_factor
        _ADAMAME_FEEDBACK_STATE['align_factor']     = align_factor

        uid2resp_lang_count = id2lang_count
        uid_order = list(dict.fromkeys(index[i] for i in range(bsz)))
        uid2first_idx = {}
        for i in range(bsz):
            if index[i] not in uid2first_idx:
                uid2first_idx[index[i]] = i
        _print_lang_dist('adamame_feedback', uid_order, uid2first_idx, uid2resp_lang_count, query_langs,
                         extra=(f'eta_batch={eta_batch:.3f} eta_ema={eta_ema:.3f} '
                                f'div_factor={diversity_factor:.3f} align_factor={align_factor:.3f}'))

        for i in range(bsz):
            uid = index[i]
            lang_i = seq_langs[i]
            query_lang = query_langs[i]
            n_lang_i = max(id2lang_count[uid][lang_i], 1)

            diversity_weight = 1.0 + (num_repeat / n_lang_i - 1.0) * diversity_factor
            matches_query = float(lang_i == query_lang and lang_i != 'unknown')
            align_weight = 1.0 + query_align_bonus * align_factor * matches_query

            scores[i] = scores[i] * diversity_weight * align_weight

        scores = _grpo_normalize(scores, index, epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_adamame_tworwd_correctonly_outcome_advantage(token_level_rewards: torch.Tensor,
                                                         eos_mask: torch.Tensor,
                                                         index: torch.Tensor,
                                                         sequences_strs,
                                                         response_strs,
                                                         query_langs,
                                                         num_repeat: int,
                                                         current_training_step: int,
                                                         total_training_steps: int,
                                                         query_align_bonus: float = 0.1,
                                                         epsilon: float = 1e-6):
    """
    Correct-only two-reward decomposition (adamame_21).

    adv_i = GRPO_norm(r_i) + B * [r_i > 0 AND lang_i == query_lang]

    Unlike tworwd, the language bonus is:
      1. Only non-zero for *correct* responses in the query language.
      2. A flat additive scalar (not GRPO-normalized), so it acts as a tiebreaker.

    This preserves the correctness gradient exactly — the GRPO normalization of
    r_i is untouched — and adds a secondary preference signal only among correct
    responses.  Wrong responses are completely unaffected by language.

    With B=0.1 (default) the language bonus is ~10% of a typical advantage unit,
    making it a tiebreaker rather than a competing objective.
    """
    detector = _build_lingua_detector()
    response_length = token_level_rewards.shape[-1]
    scores_correct = token_level_rewards.sum(dim=-1).clone().float()

    with torch.no_grad():
        bsz = scores_correct.shape[0]
        seq_langs = [_detect_lang(detector, response_strs[i]) for i in range(bsz)]

        uid2resp_lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            uid2resp_lang_count[index[i]][seq_langs[i]] += 1
        uid_order = list(dict.fromkeys(index[i] for i in range(bsz)))
        uid2first_idx = {}
        for i in range(bsz):
            if index[i] not in uid2first_idx:
                uid2first_idx[index[i]] = i
        _print_lang_dist('adamame_tworwd_correctonly', uid_order, uid2first_idx,
                         uid2resp_lang_count, query_langs,
                         extra=f'query_align_bonus={query_align_bonus}')

        # Step 1: standard GRPO normalization on correctness — identical to naive GRPO
        scores_correct = _grpo_normalize(scores_correct, index, epsilon)

        # Step 2: flat bonus only for correct responses in the query language
        raw_scores = token_level_rewards.sum(dim=-1)
        for i in range(bsz):
            lang_i = seq_langs[i]
            query_lang = query_langs[i]
            if raw_scores[i] > 0 and lang_i == query_lang and lang_i != 'unknown':
                scores_correct[i] = scores_correct[i] + query_align_bonus

        scores = scores_correct.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_adamame_allcorrect_outcome_advantage(token_level_rewards: torch.Tensor,
                                                   eos_mask: torch.Tensor,
                                                   index: torch.Tensor,
                                                   sequences_strs,
                                                   response_strs,
                                                   query_langs,
                                                   num_repeat: int,
                                                   current_training_step: int,
                                                   total_training_steps: int,
                                                   query_align_bonus: float = 1.0,
                                                   epsilon: float = 1e-6):
    """
    All-correct group language signal (adamame_23).

    Mixed-accuracy group (some correct, some wrong):
        standard GRPO on correctness rewards — language invisible.
    All-correct group (every response correct):
        GRPO on binary language-match signal {0, 1} — accuracy invisible.

    The language signal is mathematically guaranteed not to corrupt the
    accuracy gradient: it only fires when correctness provides no
    differentiation within the group.  When all responses match (or all
    mismatch) the query language, the group std collapses to ~0 and the
    epsilon floor returns zero advantage, so the model receives no gradient
    for that prompt at all — a safe no-op.
    """
    detector = _build_lingua_detector()
    response_length = token_level_rewards.shape[-1]
    raw_scores = token_level_rewards.sum(dim=-1).clone().float()

    with torch.no_grad():
        bsz = raw_scores.shape[0]
        seq_langs = [_detect_lang(detector, response_strs[i]) for i in range(bsz)]

        uid2indices = defaultdict(list)
        for i in range(bsz):
            uid2indices[index[i]].append(i)

        uid2resp_lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            uid2resp_lang_count[index[i]][seq_langs[i]] += 1
        uid_order = list(dict.fromkeys(index[i] for i in range(bsz)))
        uid2first_idx = {}
        for i in range(bsz):
            if index[i] not in uid2first_idx:
                uid2first_idx[index[i]] = i

        n_all_correct = sum(
            1 for idxs in uid2indices.values() if all(raw_scores[i] > 0 for i in idxs)
        )
        _print_lang_dist('adamame_allcorrect', uid_order, uid2first_idx,
                         uid2resp_lang_count, query_langs,
                         extra=f'all_correct_groups={n_all_correct}/{len(uid2indices)}')

        # Mixed-accuracy groups → correctness reward (standard GRPO).
        # All-correct groups → language-match reward (language GRPO).
        final_scores = raw_scores.clone()
        for uid, idxs in uid2indices.items():
            if all(raw_scores[i] > 0 for i in idxs):
                for i in idxs:
                    lang_i = seq_langs[i]
                    query_lang = query_langs[i]
                    final_scores[i] = float(lang_i == query_lang and lang_i != 'unknown')

        final_scores = _grpo_normalize(final_scores, index, epsilon)
        final_scores = final_scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return final_scores, final_scores


def compute_adamame_phased_outcome_advantage(token_level_rewards: torch.Tensor,
                                              eos_mask: torch.Tensor,
                                              index: torch.Tensor,
                                              sequences_strs,
                                              response_strs,
                                              query_langs,
                                              num_repeat: int,
                                              current_training_step: int,
                                              total_training_steps: int,
                                              query_align_bonus: float = 0.1,
                                              epsilon: float = 1e-6):
    """
    Two-phase training (adamame_24).

    Phase 1 (t/T < 0.7): pure GRPO on correctness — identical to naive GRPO.
    Phase 2 (t/T >= 0.7): correct-only tiebreaker (adamame_tworwd_correctonly).

    With total_epochs=10 this gives 7 epochs of pure accuracy training followed
    by 3 epochs of language preference shaping from a well-optimised checkpoint.
    The phase-2 language signal is identical to adamame_21 with the given B.
    """
    PHASE_SWITCH = 0.7
    phase = current_training_step / max(total_training_steps, 1)
    response_length = token_level_rewards.shape[-1]

    if phase < PHASE_SWITCH:
        scores = token_level_rewards.sum(dim=-1).clone().float()
        with torch.no_grad():
            scores = _grpo_normalize(scores, index, epsilon)
            scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
        print(f'[adamame_phased] phase=1 pure-GRPO '
              f't={current_training_step}/{total_training_steps} ratio={phase:.3f}', flush=True)
        return scores, scores
    else:
        print(f'[adamame_phased] phase=2 correctonly-tiebreaker '
              f't={current_training_step}/{total_training_steps} ratio={phase:.3f} '
              f'B={query_align_bonus}', flush=True)
        return compute_adamame_tworwd_correctonly_outcome_advantage(
            token_level_rewards, eos_mask, index,
            sequences_strs, response_strs, query_langs,
            num_repeat, current_training_step, total_training_steps,
            query_align_bonus=query_align_bonus, epsilon=epsilon,
        )


def compute_adamame_lagrangian_outcome_advantage(token_level_rewards: torch.Tensor,
                                                 eos_mask: torch.Tensor,
                                                 index: torch.Tensor,
                                                 sequences_strs,
                                                 response_strs,
                                                 query_langs,
                                                 num_repeat: int,
                                                 current_training_step: int,
                                                 total_training_steps: int,
                                                 query_align_bonus: float = 1.0,
                                                 align_target: float = 0.9,
                                                 lambda_lr: float = 0.05,
                                                 ema_alpha: float = 0.9,
                                                 epsilon: float = 1e-6):
    """
    Lagrangian alignment control (adamame_26).

    Treats language alignment as a constraint and self-tunes the bonus weight λ
    via projected gradient ascent on the Lagrange multiplier:

        m(t)    = fraction of batch responses matching query language
        m̂(t)   = β·m̂(t-1) + (1-β)·m(t)          (EMA, initialized to first batch)
        λ(t+1) = clip(λ(t) + η·(m* − m̂(t)),  0,  λ_max)

    Advantage (λ used is the value from the START of this step, before the update):
        adv_i = GRPO_norm(r_i) + λ(t)·[r_i > 0 AND lang_i == query_lang]

    The language bonus is a flat additive scalar applied only to correct responses
    in the query language — identical structure to adamame_21/24, but with a
    self-tuning weight rather than a fixed B.

    Parameters:
        query_align_bonus: λ_max — ceiling on λ (default 1.0)
        align_target:      m* — desired fraction of responses matching query lang (default 0.9)
        lambda_lr:         η — dual learning rate for the λ update (default 0.05)
        ema_alpha:         β — EMA smoothing for the match-rate signal (default 0.9)
    """
    detector = _build_lingua_detector()
    response_length = token_level_rewards.shape[-1]
    raw_scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        bsz = raw_scores.shape[0]
        seq_langs = [_detect_lang(detector, response_strs[i]) for i in range(bsz)]

        # Use λ from start of this step for the advantage computation.
        lam = _ADAMAME_LAGRANGIAN_STATE['lambda']

        scores = raw_scores.clone().float()
        scores = _grpo_normalize(scores, index, epsilon)

        for i in range(bsz):
            if raw_scores[i] > 0 and seq_langs[i] == query_langs[i] and seq_langs[i] != 'unknown':
                scores[i] = scores[i] + lam

        # --- Lagrangian update for next step ---
        n_match = sum(
            1 for i in range(bsz)
            if seq_langs[i] == query_langs[i] and seq_langs[i] != 'unknown'
        )
        m_batch = n_match / max(bsz, 1)

        if _ADAMAME_LAGRANGIAN_STATE['mhat'] is None:
            _ADAMAME_LAGRANGIAN_STATE['mhat'] = m_batch
        else:
            _ADAMAME_LAGRANGIAN_STATE['mhat'] = (
                ema_alpha * _ADAMAME_LAGRANGIAN_STATE['mhat'] + (1.0 - ema_alpha) * m_batch
            )
        mhat = _ADAMAME_LAGRANGIAN_STATE['mhat']
        new_lam = max(0.0, min(query_align_bonus, lam + lambda_lr * (align_target - mhat)))
        _ADAMAME_LAGRANGIAN_STATE['lambda'] = new_lam

        # Logging
        uid2resp_lang_count = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            uid2resp_lang_count[index[i]][seq_langs[i]] += 1
        uid_order = list(dict.fromkeys(index[i] for i in range(bsz)))
        uid2first_idx = {}
        for i in range(bsz):
            if index[i] not in uid2first_idx:
                uid2first_idx[index[i]] = i
        _print_lang_dist('adamame_lagrangian', uid_order, uid2first_idx, uid2resp_lang_count, query_langs,
                         extra=(f'm_batch={m_batch:.3f} mhat={mhat:.3f} '
                                f'lambda_used={lam:.4f} lambda_next={new_lam:.4f} '
                                f'target={align_target}'))

        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_dr_grpo_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange, max_tokens):
    """Policy loss for Dr. GRPO (https://arxiv.org/pdf/2503.20783).

    Replaces masked_mean's adaptive denominator (actual token count) with a
    constant generation budget (max_tokens), removing the length normalization
    bias where shorter responses get higher per-token gradients.

    Loss = mean_over_responses( sum_over_tokens(clipped_pg) / max_tokens )
    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped = torch.max(pg_losses, pg_losses2)

    # (bs,): sum token losses per response, divide by constant budget
    pg_loss = ((clipped * eos_mask).sum(dim=-1) / max_tokens).mean()
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
