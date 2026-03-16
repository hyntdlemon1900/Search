import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """
        后处理模型生成的响应。
        主要作用：
        1. 将 Token ID 解码为字符串。
        2. 根据停止标签（</search> 或 </answer>）截断生成内容，防止模型生成多余的幻觉内容或后续步骤。
        3. 重新编码截断后的内容为 Token ID，供后续使用。
        """
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # 核心截断逻辑：
        # - 优先检查是否有 search 动作结束标签 </search>，截断其后的所有内容。
        # - 如果没有 search，检查是否有 answer 动作结束标签 </answer>，截断其后的所有内容。
        # - 如果都没有，保留原样（可能还在思考过程中或生成不完整）。
        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        
        # 重新将处理后的字符串 Tokenize 回 ID
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest', #对齐 
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """
        更新滚动状态（Rolling State），将新生成的响应（Response）和下一步的观察（Observation）拼接到当前上下文中。
        
        Args:
            rollings: 当前的滚动状态数据（包含 input_ids, attention_mask 等）。
            cur_responses: 当前步骤模型生成的响应 Token ID。
            next_obs_ids: 环境返回的下一步观察 Token ID。
            
        Returns:
            new_rollings: 更新后的滚动状态 DataProto 对象。
        """
        # 1. 拼接：将 [历史 Context, 当前 Response, 下一步 Observation] 按顺序拼接。左填充右对齐
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # 2. 更新 Mask 和 Position IDs
        # 此时 new_attention_mask 的长度是单纯拼接后的总长度 (History + Response + Obs)，
        # 它可能已经超过了config.max_prompt_length。
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # 3. 窗口剪裁（Context Window Management）
        # 计算当前最大的有效序列长度
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        # 构造新的 DataProto
        # [:, -max_len:] 实现了滑动窗口切片：保留最右边（最新）的 max_len 个 Token，
        # 丢弃左边（最旧）的历史信息，从而保证下一轮输入的长度不会无限增长。
        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        
        # 保持元数据不变
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor,  # right_side['responses'],
                prompt_with_mask: torch.Tensor,  # right_side['responses_with_info_mask'],
                response: torch.Tensor, # cur_responses,
                info: torch.Tensor = None, # next_obs_ids,
                pad_to_left: bool = True # False
            ) -> torch.Tensor:

        pad_id = self.tokenizer.pad_token_id
        
        tensors = [prompt, response] # response历史拼接新的response
        if info is not None:
            tensors.append(info) # response历史拼接新的info
            
        concatenated = torch.cat(tensors, dim=1)
        
        if pad_to_left:
            # 左填充（用于输入 LLM）：[PAD, PAD, Content]
            mask = concatenated != pad_id 
        else:
            # 右填充（用于存储 Trajectory）：[Content, PAD, PAD]
            mask = concatenated == pad_id
            
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        
        tensors_with_mask = [prompt_with_mask, response] # response_m 历史拼接新的response
        if info is not None:
             # 对于 info 部分，我们用相同 shape 的 pad_id 去填满
             info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
             tensors_with_mask.append(info_mask)
             
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)       
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info # 所以这里padded_tensor带info, padded_tensor_with_info带info_pad

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """
        更新“右侧”状态（即生成部分）。
        与 _update_rolling_state 不同，right_side 仅保存生成的 Response 和 Observation，不包含最初的 Prompt。
        这部分最终用于拼接成完整的 Trajectory。
        """
        if next_obs_ids != None:
            # 如果有观察结果（比如搜索后），则拼接 [Response, Masked_Observation]
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            # 如果没有观察结果（比如只由 LLM 生成了部分），则只拼接 [Response]
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        
        # 同样进行长度管理，防止显存溢出，但通常这里主要关注生成部分的累积
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
        处理多 GPU 生成时的 Padding 问题。
        - vLLM 或其他分布式推理框架通常要求 Batch Size 能被 GPU 数量整除。
        - 该函数会自动检测并填充 Batch，使其大小对齐，生成后再移除填充部分。
        """
        num_gpus = self.config.num_gpus
        
        # 1. 如果是单卡或不需要 Padding，直接生成
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        # 统一转为 long 类型，避免类型不匹配
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
            
        # 2. 如果恰好整除，也可以直接生成
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # 3. 需要填充的情况
        # 计算需要补充多少个样本
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        # 构造填充数据：复制第一个样本来填充
        # 外层循环遍历的是数据字段（如 'input_ids', 'attention_mask'），而不是样本
        for k, v in active_batch.batch.items():
            # v 是一个 Tensor，形状通常是 (Batch_Size, Sequence_Length)
            # 取第一条样本数据 [0:1] 重复 padding_size 次，生成 pad_sequence
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            
            # 在 dim=0 (样本/Batch维度) 上拼接，相当于增加了 Batch Size
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # 4. 执行生成（此时 Batch Size 已对齐）
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # 5. 后处理：移除填充的样本
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # 如果 output 包含 meta_info，也需要对应裁剪
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size] # Tensor 类型此时通常是 batch 维度的，需要切片
                else:
                    trimmed_meta[k] = v # 非 Tensor 类型（如 list 或 scalar）可能不需要切片，或者需要具体分析，这里假设保持原样
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        # 从原始输入的右侧（末尾）截取最新的 max_start_length 1024 个 token 作为 Prompt。
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]} # 一个带info，一个不带info
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool) # 一个布尔张量，记录了当前 Batch 中哪些样本还在进行对话（即尚未结束）。True 表示该样本还在继续，False 表示已经结束（如已生成最终答案）。
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        # 主生成循环 (ReAct Logic: Think/Act -> Search -> Observe -> Loop)
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            # 裁剪 Context 到最大有效长度，提升推理效率
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids'] # 这些数据是右对齐的
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({ # 重新构建一个新的 DataProto 对象，其中只保留那些状态为活跃（Active）的样本数据
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active) # self.actor_rollout_wg.generate_sequences(active_batch) 生成长度对齐到500，但基于rollings_active

            meta_info = gen_output.meta_info            
            # 处理后的 responses_ids 长度不再是500
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses']) # 根据停止标签（</search> 或 </answer>）截断生成内容，基于rollings_active
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask) # 将基于rollings_active的responses还原到完整的 Batch 维度。

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions( # 检索/回答动作执行后，环境返回的下一个观察（Observation）以及该样本是否结束（Done）等信息
                responses_str, self.tokenizer.pad_token, active_mask
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask # 更新掩码，done的样本从active样本中剔除
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs) # tokenize操作，对齐，长度不超过config.max_obs_length
            
            # Update states
            rollings = self._update_rolling_state( # 拼接，长度通过滑动窗口控制到不超过self.config.max_prompt_length
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids, # 可能有pad
                next_obs_ids # 可能有pad
            )
            
        # final LLM rollout
        # 为什么在循环结束后还需要执行一次生成逻辑？
        # 1. 消费最后的 Observation：
        #    主循环 (Main generation loop) 的最后一步通常是执行 Action (Search) 并将 Observation (next_obs_ids) 更新到上下文 (rollings) 中。
        #    此时，模型实际上拥有了最新的环境反馈，但还没有基于这个反馈进行最后一步的生成（Reasoning/Answer）。
        # 2. 收尾/强制结束：
        #    当达到最大交互轮数 (max_turns) 后，我们给模型最后一次机会根据收集到的所有信息生成最终结论。
        #    这就通过 do_search=False 来强制模型不再发起新的搜索，而是尝试给出 Answer。
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            # 注意：这里的 do_search=False，意味着即使模型输出 <search> 也不会执行搜索，防止在最后一步陷入死循环。
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'], # 右对齐
            right_side['responses'] # 左对齐但info没被pad
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']), # 右对齐
            self.tensor_fn.create_attention_mask(final_output['responses']) # 左对齐但info没被pad
        ], dim=1)
        final_output['info_mask'] = torch.cat([ # info被pad
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        在多个环境中执行模型预测的动作（Action）。
        这个函数相当于 Reinforcement Learning 中的 `env.step()`。
        
        功能流程：
        1. 解析模型输出，提取动作类型（Search/Answer）和内容。
        2. 批量执行搜索（如果动作是 Search）。
        3. 构造下一个观察（Observation）或反馈信息。
        
        Args:
            predictions: 模型生成的预测文本列表。
            pad_token: 用于 Padding 的 Token（此处未实际使用）。
            active_mask: 标记当前仍在活跃的样本。
            do_search: 是否执行实际的搜索操作。
            
        Returns:
            next_obs: 环境反馈的观察结果列表。
            dones: 标记该样本是否结束（1为结束，0为未结束）。
            valid_action: 标记动作格式是否有效。
            is_search: 标记是否执行了搜索动作。
        """
        # 1. 后处理：从文本中解析出 Action列表 (search/answer) 和 Content列表 (query/answer text)
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        # 2. 批量搜索逻辑
        # 收集所有 Search 动作的 Query
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        
        # 执行批量搜索请求
        if do_search:
            search_results = self.batch_search(search_queries)
            # 校验：返回结果数量应与请求数量一致
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            # 如果不执行搜索（如验证模式），填充空结果
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        # 3. 遍历每个样本，根据动作类型构造反馈
        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            # Case A: 样本本身已非活跃（上一轮已结束），直接 Pass
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                # Case B: 动作为 Answer -> 任务完成
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1) # 标记结束
                    valid_action.append(1)
                    is_search.append(0)
                
                # Case C: 动作为 Search -> 返回搜索结果
                elif action == 'search':
                    # 取出对应的搜索结果，包裹在 <information> 标签中
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0) # 搜索后需继续生成，未结束
                    valid_action.append(1)
                    is_search.append(1)
                
                # Case D: 既不是search也不是answer -> 返回提示 Prompt 要采取动作
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0) # 错误后允许重试，未结束
                    valid_action.append(0)
                    is_search.append(0)
            
        # 确保所有搜索结果都被消耗完
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[str]]:
        """
        解析 LLM 的预测输出，提取动作和内容。
        
        Args:
            predictions: LLM 生成的原始文本列表。
            
        Returns:
            Tuple (动作列表, 内容列表)
            - actions: ['search', 'answer', None, ...]
            - contents: [query_string, answer_string, '', ...]
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                # 使用正则表达式匹配 <search>...</search> 或 <answer>...</answer>
                # \1 表示引用第一个捕获组的内容，确保标签闭合匹配（即 <search> 配 </search>）
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                
                if match:
                    # 匹配成功，提取动作类型和标签内的内容
                    content = match.group(2).strip()  # 标签内部的文本
                    action = match.group(1)           # 动作类型 (search 或 answer)
                else:
                    # 匹配失败（格式错误或未完成）
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
