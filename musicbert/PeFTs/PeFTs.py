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
                # Create unique VeraLinear layer with its own A and B
                in_features = child.in_features
                
                # 1. Create the random tensors
                vera_A_tensor = torch.empty(rank, in_features, device=child.weight.device)
                vera_B_tensor = torch.empty(in_features, rank, device=child.weight.device)

                # 2. Initialize them with Kaiming Uniform
                torch.nn.init.kaiming_uniform_(vera_A_tensor, a=math.sqrt(5))
                torch.nn.init.kaiming_uniform_(vera_B_tensor, a=math.sqrt(5))

                # 3. Place them in ParameterDicts as required by VeraLinear
                vera_A_dict = nn.ParameterDict({
                    "default": nn.Parameter(vera_A_tensor, requires_grad=False)
                })
                vera_B_dict = nn.ParameterDict({
                    "default": nn.Parameter(vera_B_tensor, requires_grad=False)
                })

                # 4. Create the VeraLinear layer with its own unique A and B
                vera_layer = VeraLinear(
                    base_layer=child,
                    adapter_name="default",
                    r=rank,
                    vera_dropout=dropout,
                    fan_in_fan_out=False,
                    vera_A=vera_A_dict,
                    vera_B=vera_B_dict,
                )
                setattr(module, child_name, vera_layer)
                replacements_made += 1
        
        # Recurse into child modules
        replacements_made += inject_vera(child, rank, dropout, 
                                         target_modules, current_path)
    
    return replacements_made