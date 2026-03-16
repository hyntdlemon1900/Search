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

import re
import string
import random

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text):
    """
    检查生成的文本是否符合预定义的格式序列。
    预期的格式包括：
    1. 必须以 "<|im_start|>assistant" 开头。
    2. 标签必须成对出现且闭合。
    3. 标签的顺序必须遵循特定的状态机流转：
       start -> <think> -> ... -> </think> -> <search> -> ... -> </search> -> <information> -> ... -> </information> -> <think> ... -> <answer> -> ... -> </answer> -> end
    """
    # 查找 "<|im_start|>assistant" 的位置，允许后面有空白字符
    # Update: check for 'Assistant:' as well regarding the base model
    assistant_pattern = r"(<\|im_start\|>assistant|Assistant:)\s*"
    assistant_match = re.search(assistant_pattern, text, re.IGNORECASE)
    
    if not assistant_match:
        return False, "Missing assistant marker"
    
    # 提取 assistant 标记之后的内容
    start_pos = assistant_match.end()
    content = text[start_pos:]
    
    # Check for balanced tags
    # 检查标签是否平衡（开标签和闭标签数量一致）
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # 现在检查正确的序列模式，并确保没有多余的内容

    # 1. 首先根据我们识别的任何标签分割内容
    # 使用捕获组 () 保留分隔符（即标签本身）
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. 跟踪预期序列中的当前状态
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end
    
    # 3. 检查每一部分（标签或内容）
    for i, part in enumerate(parts):
        # Skip empty parts
        # 跳过空白部分
        if not part.strip():
            continue
            
        # 检查这部分是否是一个标签
        if re.match(r"</?(?:think|search|information|answer)>", part):
            # 这是一个标签，检查它在当前状态下是否合法

            # <think> 可以在开始时出现，或者在 <information> 块结束后出现（进入下一轮思考）
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            # </think> 只能在 <think> 内部出现，标志思考结束
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            # <search> 只能在思考结束后出现（思考决定去搜索）
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            # </search> 只能在搜索内部出现
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            # <information> 只能在搜索结束后出现（返回搜索结果）
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            # </information> 只能在信息块内部出现
            elif part == "</information>" and state == "in_information":
                state = "information"
            # <answer> 只能在思考结束后出现（最终得出答案）
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            # </answer> 只能在答案内部出现，标志结束
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # 这是内容文本，检查它在当前状态下是否允许存在
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # 在标签对内部，允许存在内容
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                # 在标签之间（例如 think 结束和 search 开始之间），只允许空白
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    # 检查最终状态，必须是 "end"（即成功闭合了 answer）
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em(solution_str, ground_truth, method='strict', structure_format_score=0, final_format_score=0, retrieval_score=0, format_score=0, score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
            
    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score # 0.3
            else:
                return structure_format_score # 0.2
        else:
            return 0
    else:
        if em_check(answer, ground_truth['target']):
            if is_valid_format:
                return score # 1
            else:
                return score - structure_format_score # 0.8
        elif is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score # 0.3
            else:
                return structure_format_score # 0.2
        else:
            return final_format_score # 0.1
