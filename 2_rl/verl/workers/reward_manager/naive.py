# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
import os
import re

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch

try:
    from lingua import Language, LanguageDetectorBuilder
    _lingua_detector = LanguageDetectorBuilder.from_languages(
        Language.FRENCH, Language.JAPANESE, Language.KOREAN, Language.THAI, Language.PORTUGUESE,
        Language.ARABIC, Language.ENGLISH, Language.SPANISH, Language.VIETNAMESE, Language.CHINESE,
        Language.BENGALI, Language.SWAHILI, Language.TELUGU, Language.GERMAN,
    ).build()
except ImportError:
    _lingua_detector = None


def _extract_think_content(text):
    """Extract the content inside the first <think>...</think> block."""
    m = re.search(r'<think>([\s\S]+?)</think>', text)
    return m.group(1).strip() if m else None


def _detect_language(text):
    """Return ISO 639-1 language code, or 'unknown' on failure."""
    if not text or len(text.split()) < 3:
        return 'unknown'
    try:
        lang = _lingua_detector.detect_language_of(text)
        return lang.iso_code_639_1.name.lower() if lang else 'unknown'
    except Exception:
        return 'unknown'


class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, split='train') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.split = split
        self.compute_score = compute_score or _default_compute_score
        self._batch_idx = 0
        lang_log_path = os.environ.get(
            'LANG_LOG_FILE',
            os.path.join('logs', 'lang_dist.jsonl'),
        )
        os.makedirs(os.path.dirname(lang_log_path) or '.', exist_ok=True)
        self._lang_log_file = open(lang_log_path, 'a', buffering=1)  # line-buffered

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores'], [], []

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        sequences_strs = []
        response_strs = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            sequences_strs.append(sequences_str)
            response_strs.append(self.tokenizer.decode(valid_response_ids))

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            # acc ∈ {0,1} + format ∈ {0,1}
            score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
            )

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                n = 300
                if len(sequences_str) > 2 * n:
                    print(sequences_str[:n] + '\n[...]\n' + sequences_str[-n:])
                else:
                    print(sequences_str)

        # Log one record per prompt: query_lang (single) + think_lang counter over all n rollouts
        # Items are ordered [prompt0]*n, [prompt1]*n, ... (interleave=True repeat)
        from collections import defaultdict, Counter
        prompts = defaultdict(lambda: {'query_lang': None, 'think_lang': Counter()})
        for i, seq in enumerate(sequences_strs):
            data_item = data[i]
            prompt_idx = data_item.non_tensor_batch.get('index', i)
            entry = prompts[int(prompt_idx)]

            if entry['query_lang'] is None:
                lang = data_item.non_tensor_batch.get('lang', 'unknown')
                if isinstance(lang, bytes):
                    lang = lang.decode()
                entry['query_lang'] = str(lang)

            think_text = _extract_think_content(seq)
            entry['think_lang'][_detect_language(think_text)] += 1

        for prompt_idx, entry in prompts.items():
            record = {
                'batch': self._batch_idx,
                'split': self.split,
                'prompt_idx': prompt_idx,
                'query_lang': entry['query_lang'],
                'think_lang': dict(entry['think_lang']),
            }
            self._lang_log_file.write(json.dumps(record) + '\n')
        self._batch_idx += 1

        return reward_tensor, sequences_strs, response_strs
