import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass
from typing import List, Union
import numpy as np

@dataclass
class TensorConfig:
    pad_token_id: int
    max_prompt_length: int
    max_obs_length: int
    max_start_length: int

class TensorHelper:
    def __init__(self, config: TensorConfig):
        self.config = config

    def cut_to_effective_len(self, tensor_dict: Dict[str, torch.Tensor], 
                            batch_keys: List[str], cut_left: bool = True) -> Dict[str, torch.Tensor]:
        """Cut tensors to their effective length based on attention mask."""
        effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
        result = tensor_dict.copy()

        for key in batch_keys:
            if cut_left:
                result[key] = tensor_dict[key][:,-effective_len:]
            else:
                result[key] = tensor_dicts[key][:effective_len]
        # for key in non_tensor_batch_keys:_example_level_pad
        #     result[non_tensor_batch_keys][key] = tensor_dict[non_tensor_batch_keys][key]
        # for key in meta_info_keys:non_tensor_batch_keys:List[str],meta_info_keys: List[str],
        #     result[meta_info_keys][key] = tensor_dict[meta_info_keys][key]

        return result

    def convert_pad_structure(self, tensor: torch.Tensor, pad_to_left: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert padding structure and return sorted tensor with indices."""
        mask = tensor != self.config.pad_token_id if pad_to_left else tensor == self.config.pad_token_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        return tensor.gather(1, sorted_indices), sorted_indices

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input ids."""
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create position ids from attention mask."""
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(self, tensors: List[torch.Tensor], 
                               pad_to_left: bool = True) -> torch.Tensor:
        """Concatenate tensors and handle padding."""
        concatenated = torch.cat(tensors, dim=1)
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    def concatenate_with_padding1(self, tensors: List[Union[torch.Tensor, np.ndarray]], 
                                pad_to_left: bool = True) -> np.ndarray:
        # 获取batch_size
        first_tensor = tensors[0]
        if isinstance(first_tensor, np.ndarray) and first_tensor.dtype == np.object_:
            batch_size = len(first_tensor)
        else:
            batch_size = first_tensor.shape[0] if hasattr(first_tensor, 'shape') and len(first_tensor.shape) > 0 else 1
        
        result_batch = []
        
        for i in range(batch_size):
            combined_tokens = []
            
            for tensor in tensors:
                if isinstance(tensor, np.ndarray) and tensor.dtype == np.object_:
                    sample_tokens = tensor[i] if i < len(tensor) else []
                elif isinstance(tensor, torch.Tensor):
                    sample_tokens = tensor[i].cpu().numpy().tolist() if i < tensor.size(0) else []
                elif isinstance(tensor, np.ndarray):
                    sample_tokens = tensor[i].tolist() if i < tensor.shape[0] else []
                else:
                    sample_tokens = tensor[i] if i < len(tensor) else []

                combined_tokens.extend(sample_tokens)
            
            # 修正：将列表转换为tensor后再处理
            combined_tensor = torch.tensor([combined_tokens], dtype=torch.long)  # 添加batch维度
            padded_tensor, _ = self.convert_pad_structure(combined_tensor, pad_to_left)
            result_batch.append(padded_tensor.squeeze(0).cpu().numpy())  # 移除batch维度并转numpy
        
        return np.array(result_batch, dtype=object)

    def _example_level_pad(self, responses: torch.Tensor, 
                          responses_str: List[str], 
                          active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        Pad responses for non-active examples with pad tokens.
        """
        # print(active_mask.sum())
        # print(responses.shape[0])
        assert active_mask.sum() == responses.shape[0]
        # Create masked responses tensor

        batch_size = active_mask.shape[0]
        seq_len = responses.shape[1]
        padded_responses = torch.full(
            (batch_size, seq_len), self.config.pad_token_id,
            dtype=responses.dtype, device=responses.device
        )
        padded_responses[active_mask] = responses
        
        # Create masked response strings
        padded_responses_str = [""] * batch_size
        
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                s += 1
        
        return padded_responses, padded_responses_str
    