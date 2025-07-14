import math
from typing import Sequence
import torch
import torch.nn as nn
from peft.tuners.lora import Linear as LoraLinear
from peft.tuners.vera import Linear as VeraLinear


def inject_lora(
    module: nn.Module,
    rank: int,
    alpha: int,
    dropout: float,
    rslora: bool,
    dora: bool,
    target_modules: Sequence[str],
    module_path: str = "",
):
    replacements_made = 0
    
    
    for child_name, child in list(module.named_children()):
        current_path = f"{module_path}.{child_name}" if module_path else child_name
        
        # Check if this linear layer should be wrapped
        if isinstance(child, nn.Linear):
            
            # Match either exact name or if any target_module is a suffix of the path
            should_wrap = (
                child_name in target_modules or
                any(current_path.endswith(target) for target in target_modules)
            )
            
            if should_wrap:
                lora_layer = LoraLinear(
                    base_layer=child,
                    adapter_name="default",
                    r=rank,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    use_rslora=rslora,
                    use_dora=dora,
                    fan_in_fan_out=False,
                )
                setattr(module, child_name, lora_layer)
                replacements_made += 1
        
        # Recurse into child modules
        replacements_made += inject_lora(child, rank, alpha, dropout,rslora, dora, 
                                         target_modules, current_path)
    
    return replacements_made

def inject_vera(
    module: nn.Module,
    rank: int,
    dropout: float,
    vera_A: nn.ParameterDict,
    vera_B: nn.ParameterDict,
    target_modules: Sequence[str],
    module_path: str = ""
):
    replacements_made = 0
    
    
    for child_name, child in list(module.named_children()):
        current_path = f"{module_path}.{child_name}" if module_path else child_name
        
        # Check if this linear layer should be wrapped
        if isinstance(child, nn.Linear):
            # Match either exact name or if any target_module is a suffix of the path
            should_wrap = (
                child_name in target_modules or
                any(current_path.endswith(target) for target in target_modules)
            )
            
            if should_wrap:
                vera_layer = VeraLinear(
                    base_layer=child,
                    adapter_name="default",
                    r=rank,
                    vera_dropout=dropout,
                    vera_A=vera_A,
                    vera_B=vera_B,
                    fan_in_fan_out=False
                )
                setattr(module, child_name, vera_layer)
                replacements_made += 1
        
        # Recurse into child modules
        replacements_made += inject_vera(child, rank, dropout, vera_A, vera_B,
                                         target_modules, current_path)
    
    return replacements_made