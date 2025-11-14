"""
(Triantafulloy)
Patch Fairseq's Optimizer to include PeFT parameters that may be missed by the default optimizer builder.
In most cases, Fairseq's default optimizer builder works fine
"""


import logging
from fairseq.trainer import Trainer

LOGGER = logging.getLogger(__name__)

# (Triantafulloy) Added this flag to control whether to use custom optimizer or not
_USE_CUSTOM_OPTIMIZER = False
def set_use_custom_optimizer(use_custom_optimizer: bool):
    global _USE_CUSTOM_OPTIMIZER
    _USE_CUSTOM_OPTIMIZER = use_custom_optimizer

    if _USE_CUSTOM_OPTIMIZER:
        LOGGER.info("Enabling custom optimizer - applying patch.")
        patch_trainer()
    else:
        LOGGER.info("Custom optimizer disabled - using Fairseq default.")

def patch_trainer():

    # Check if the Trainer has already been patched
    if getattr(Trainer, "_pefts_patched", False):
        LOGGER.debug("Fairseq Trainer already patched for PeFTs.")
        return
    
    LOGGER.info("Applying PeFTs patch to Fairseq Trainer...")
    Trainer._pefts_patched = True

    # Save the original
    Trainer._orig_build_optimizer = Trainer._build_optimizer

    def _build_optimizer_with_pefts(self):
        LOGGER.info("Custom _build_optimizer_with_pefts CALLED.")

        # Run Fairseq’s stock builder
        self._orig_build_optimizer()

        # Grab the FairseqOptimizer wrapper
        fairseq_optim = self._optimizer

        # Unwrap to the raw torch Optimizer
        torch_optim = getattr(fairseq_optim, "optimizer", fairseq_optim)

        # Collect all LoRA parameters
        lora_params = [
            p for n, p in self.model.named_parameters()
            if "lora_" in n and p.requires_grad
        ]

        if lora_params:
            LOGGER.info(f"Adding {len(lora_params)} LoRA/DoRA parameters to optimizer group.")
            # Check if these params are already in an optimizer group
            existing_params_in_group0 = set(id(p) for p in torch_optim.param_groups[0]['params'])
            new_lora_params = [p for p in lora_params if id(p) not in existing_params_in_group0]

            if new_lora_params:
                torch_optim.param_groups[0]["params"].extend(new_lora_params)
                LOGGER.info(f"Added {len(new_lora_params)} new LoRA/DoRA parameters to optimizer.")
            else:
                LOGGER.info("LoRA/DoRA parameters already in the optimizer group.")

        # Collect all VeRA parameters
        vera_params = [
            p for n, p in self.model.named_parameters()
            if "vera_" in n and p.requires_grad
        ]

        if vera_params:
            LOGGER.info(f"Adding {len(vera_params)} VeRA parameters to optimizer group.")
            existing_params_in_group0 = set(id(p) for p in torch_optim.param_groups[0]['params'])
            new_vera_params = [p for p in vera_params if id(p) not in existing_params_in_group0]

            if new_vera_params:
                torch_optim.param_groups[0]["params"].extend(new_vera_params)
                LOGGER.info(f"Added {len(new_vera_params)} new VeRA parameters to optimizer.")
            else:
                LOGGER.info("VeRA parameters already in the optimizer group.")

        # Collect all SVFT parameters
        svft_params = [
            p for n, p in self.model.named_parameters()
            if "svft_" in n and p.requires_grad
        ]

        if svft_params:
            LOGGER.info(f"Adding {len(svft_params)} SVFT parameters to optimizer group.")
            existing_params_in_group0 = set(id(p) for p in torch_optim.param_groups[0]['params'])
            new_svft_params = [p for p in svft_params if id(p) not in existing_params_in_group0]

            if new_svft_params:
                torch_optim.param_groups[0]["params"].extend(new_svft_params)
                LOGGER.info(f"Added {len(new_svft_params)} new SVFT parameters to optimizer.")
            else:
                LOGGER.info("SVFT parameters already in the optimizer group.")

        # Collect all MuMoE parameters
        mumoe_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad
        ]
        
        if mumoe_params:
            LOGGER.info(f"Adding {len(mumoe_params)} MuMoE parameters to optimizer group.")
            existing_params_in_group0 = set(id(p) for p in torch_optim.param_groups[0]['params'])
            new_mumoe_params = [p for p in mumoe_params if id(p) not in existing_params_in_group0]

            if new_mumoe_params:
                torch_optim.param_groups[0]["params"].extend(new_mumoe_params)
                LOGGER.info(f"Added {len(new_mumoe_params)} new MuMoE parameters to optimizer.")
            else:
                LOGGER.info("MuMoE parameters already in the optimizer group.")

    # Override Fairseq’s builder
    Trainer._build_optimizer = _build_optimizer_with_pefts
    LOGGER.info("Fairseq Trainer patch to use PeFTs applied successfully.")