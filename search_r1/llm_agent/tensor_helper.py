import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class TensorConfig:
    pad_token_id: int
    max_prompt_length: int
    max_obs_length: int
    max_start_length: int

class TensorHelper:
    """
    张量处理辅助类：负责生成过程中的张量操作，包括剪裁、填充、掩码生成等。
    主要用于处理变长序列在批处理时的对齐和 Mask 操作。
    """
    def __init__(self, config: TensorConfig):
        self.config = config

    def cut_to_effective_len(self, tensor_dict: Dict[str, torch.Tensor], 
                            keys: List[str]) -> Dict[str, torch.Tensor]: # ['input_ids', 'attention_mask', 'position_ids']
        """
        根据 attention_mask 将张量剪裁到最大有效长度。
        - 移除批次中所有样本都为 padding 的多余部分。
        - 通常用于处理 Transformer 的输入，减少计算量。
        """
        effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
        result = tensor_dict.copy()
        
        for key in keys:
            result[key] = tensor_dict[key][:, -effective_len:]
        return result # 取prompt/obs/response部分的有效长度进行剪裁，去掉前面多余的 pad token

    def convert_pad_structure(self, tensor: torch.Tensor, pad_to_left: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        转换填充结构并返回排序后的张量与索引。
        - 用于将分散的 pad token 聚集到一侧（左侧或右侧）。
        - 使用 argsort 对非 pad 元素进行重排。
        """
        # 1. 构建 mask：用于区分 Pad Token 和有效内容
        # 如果 pad_to_left=True (左填充)：非 Pad 为 1 (排在后)，Pad 为 0 (排在前)。
        # 如果 pad_to_left=False (右填充)：Pad 为 1 (排在后)，非 Pad 为 0 (排在前)。
        mask = tensor != self.config.pad_token_id if pad_to_left else tensor == self.config.pad_token_id
        
        # 2. 利用排序实现移动：
        # argsort 会将值为 0 的元素索引排在值为 1 的元素索引前面。
        # stable=True 至关重要：它保证了原本顺序相邻的有效 Token 在排序后依然保持相对顺序（不会打乱句子文本）。
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        
        # 3. 重组 Tensor：
        # gather 按照 sorted_indices 的指示，将原始 Tensor 的元素搬运到新位置，从而实现“将 Pad 挤到一侧”的效果。
        return tensor.gather(1, sorted_indices), sorted_indices

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        根据 input_ids 创建 attention_mask。
        - pad_token 处为 0，其余为 1。
        """
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        根据 attention_mask 创建 position_ids。
        - 仅对有效 token 进行计数，计算其位置索引。
        """
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(self, tensors: List[torch.Tensor], 
                               pad_to_left: bool = True) -> torch.Tensor:
        """
        连接多个张量并处理填充。
        - 将多个张量拼接后，重新整理 padding 位置，确保对齐。
        """
        # 1. 物理拼接：直接在序列维度 (dim=1) 上拼接。
        # 结果张量的长度 = 所有输入张量长度之和。
        # 例如：Tensor A (Batch, 10) + Tensor B (Batch, 5) -> concatenated (Batch, 15)
        # 注意：此时 Pad Token 可能散落在中间，例如 [Pad, Pad, Text, Pad, Text]
        concatenated = torch.cat(tensors, dim=1)
        
        # 2. 逻辑重排：调用 convert_pad_structure 将所有的 Pad 移到一侧。
        # 此操作不会改变张量的形状或总长度，只是改变了行内元素的顺序。
        # 结果：[Pad, Pad, Pad, Text, Text] (若 pad_to_left=True)
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        
        return padded_tensor

    def _example_level_pad(self, responses: torch.Tensor, 
                          responses_str: List[str], 
                          active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        对已结束对话（非活跃）的样本进行全填充（Masking）。
        该方法将当前仍在生成的 responses（仅包含 active 的部分）还原到完整的 Batch 维度。
        
        Args:
            responses: 仅包含 active 样本的生成结果，Tensor 形状为 (num_active, seq_len)。
            responses_str: 仅包含 active 样本的生成字符串列表。
            active_mask: 完整 Batch 的活跃状态掩码，shape 为 (batch_size,)。
            
        Returns:
            padded_responses: 扩充后的完整 Batch Tensor，形状 (batch_size, seq_len)。
                              非 active 位置全填 pad_token_id。
            padded_responses_str: 扩充后的完整字符串列表，非 active 位置为空字符串 ""。
        """
        # 校验：输入的 responses 数量必须等于 active 状态的数量
        assert active_mask.sum() == responses.shape[0]
        
        batch_size = active_mask.shape[0]
        seq_len = responses.shape[1]
        
        # 1. 初始化全 Pad 的 Tensor
        padded_responses = torch.full(
            (batch_size, seq_len), self.config.pad_token_id,
            dtype=responses.dtype, device=responses.device
        )
        # 2. 将 active 的生成结果填入对应位置
        padded_responses[active_mask] = responses
        
        # 3. 初始化全空的字符串列表
        padded_responses_str = [""] * batch_size
        
        # 4. 遍历 active_mask，将有效字符串填回对应位置
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                s += 1
                
        return padded_responses, padded_responses_str