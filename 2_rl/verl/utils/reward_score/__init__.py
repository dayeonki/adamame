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
# from . import gsm8k, math, prime_math, prime_code


def _check_think_format(solution_str):
    """Binary format check: 1.0 if both conditions hold, else 0.0.

    Conditions:
      1. <think>...</think> block exists and contains at least 10 characters.
      2. \\boxed{} appears in the response portion (after </think>).
    """
    import re
    match = re.search(r'<think>([\s\S]+?)</think>([\s\S]+)', solution_str)
    if not match:
        return 0.0
    think_content = match.group(1).strip()
    response_content = match.group(2)
    has_think = len(think_content) >= 10
    has_boxed = bool(re.search(r'\\boxed\{', response_content))
    return 1.0 if (has_think and has_boxed) else 0.0


def _default_compute_score(data_source, solution_str, ground_truth, use_format_reward=True):
    if data_source in ('openai/gsm8k', 'zoey/gsm8k'):
        from math_verify import parse, verify
        try:
            parsed = parse(solution_str)
            acc = float(verify(parsed, parse(ground_truth)))
        except Exception:
            acc = 0.0
        res = acc + ((_check_think_format(solution_str)) if use_format_reward else 0.0)  # acc ∈ {0,1} [+ format ∈ {0,1}]
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    else:
        raise NotImplementedError

    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
