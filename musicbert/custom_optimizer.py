import logging
import torch
import ast # For safely evaluating string representations of tuples/lists
from collections import defaultdict # Might be needed by FP16Optimizer internals, good to have if copying parts
from itertools import chain # Might be needed by FP16Optimizer internals

from fairseq import utils # For utils.has_parameters
from fairseq.optim import FairseqOptimizer # Base class for our custom wrapper
from fairseq.optim import lr_scheduler # For building the LR scheduler
from fairseq.optim.fp16_optimizer import FP16Optimizer 

LOGGER = logging.getLogger(__name__)

# This map helps in creating the correct PyTorch optimizer instance for the FP32 optimizer
OPTIMIZER_REGISTRY = {
    "adam": torch.optim.AdamW,
    # Add other optimizers if you use them, e.g., "sgd": torch.optim.SGD
}

# --- Custom Fairseq Optimizer Wrapper (This will wrap the FP32 torch.optim.Optimizer) ---
class CustomPeftFairseqOptimizer(FairseqOptimizer):
    def __init__(self, cfg_optimizer_args, torch_fp32_optimizer_instance):
        super().__init__(cfg_optimizer_args) # cfg_optimizer_args is self.cfg.optimizer from Trainer
                                             # Base FairseqOptimizer stores this as self.cfg
        self._optimizer = torch_fp32_optimizer_instance # This now optimizes FP32 (master) params

        # These will be populated by the _build_custom_peft_optimizer method after this instance is created
        self.initial_lr_for_base_group = None
        self.initial_lr_for_lora_group = None
        self.initial_lr_for_vera_group = None
        
        self.lora_lr_schedule_type = 'proportional' # Default, will be overwritten
        self.vera_lr_schedule_type = 'proportional' # Default, will be overwritten

    @property
    def optimizer(self):
        # This getter allows Fairseq components to access the underlying torch optimizer
        return self._optimizer

    @property
    def param_groups(self):
        # Delegates to the wrapped PyTorch optimizer's param_groups
        return self._optimizer.param_groups if self._optimizer else []

    def get_lr(self):
        # Fairseq's main scheduler often uses the LR of the first group as 'the' LR
        if not self.param_groups: # Use the property
            return 0.0
        return self.param_groups[0]['lr']

    def set_lr(self, new_base_lr_from_scheduler):
        if not self._optimizer or not hasattr(self._optimizer, 'param_groups'):
            LOGGER.error("CustomPeftFairseqOptimizer: self._optimizer not set or has no param_groups.")
            return

        if self.initial_lr_for_base_group is None:
            LOGGER.warning("CustomPeftFairseqOptimizer: Initial LRs for groups not fully set for differential scheduling. Applying LR uniformly.")
            for group in self.param_groups: # Use the property
                group['lr'] = new_base_lr_from_scheduler
            return

        current_optimizer_steps = 0
        if self._optimizer.state: 
            try:
                current_optimizer_steps = next(iter(self._optimizer.state.values())).get('step', torch.tensor(0)).item()
            except: pass 

        for group in self.param_groups: # Use the property
            group_type = group.get('group_type', 'base')

            if group_type == 'base':
                group['lr'] = new_base_lr_from_scheduler
            
            elif group_type == 'lora':
                if self.initial_lr_for_lora_group is None:
                    group['lr'] = new_base_lr_from_scheduler; continue
                if self.lora_lr_schedule_type == 'fixed':
                    group['lr'] = self.initial_lr_for_lora_group
                elif self.lora_lr_schedule_type == 'proportional':
                    main_cli_initial_lr = self.cfg.lr[0] # self.cfg is cfg_optimizer_args from __init__
                    if main_cli_initial_lr > 1e-9: 
                        scale_factor = new_base_lr_from_scheduler / main_cli_initial_lr
                        group['lr'] = scale_factor * self.initial_lr_for_lora_group
                    else: 
                        group['lr'] = 0.0 if new_base_lr_from_scheduler == 0.0 else self.initial_lr_for_lora_group
                else: 
                    LOGGER.warning(f"Unknown lora_lr_schedule_type: {self.lora_lr_schedule_type}. Applying main scheduled LR.")
                    group['lr'] = new_base_lr_from_scheduler
            
            elif group_type == 'vera':
                if self.initial_lr_for_vera_group is None:
                    group['lr'] = new_base_lr_from_scheduler; continue
                if self.vera_lr_schedule_type == 'fixed':
                    group['lr'] = self.initial_lr_for_vera_group
                elif self.vera_lr_schedule_type == 'proportional':
                    main_cli_initial_lr = self.cfg.lr[0] # self.cfg is cfg_optimizer_args
                    if main_cli_initial_lr > 1e-9: 
                        scale_factor = new_base_lr_from_scheduler / main_cli_initial_lr
                        group['lr'] = scale_factor * self.initial_lr_for_vera_group
                    else: 
                        group['lr'] = 0.0 if new_base_lr_from_scheduler == 0.0 else self.initial_lr_for_vera_group
                else: 
                    LOGGER.warning(f"Unknown vera_lr_schedule_type: {self.vera_lr_schedule_type}. Applying main scheduled LR.")
                    group['lr'] = new_base_lr_from_scheduler
            else: 
                if group_type not in ['base', 'lora', 'vera']: LOGGER.debug(f"Group type {group_type} falling back to base LR.")
                group['lr'] = new_base_lr_from_scheduler
        
        if current_optimizer_steps > 0 and current_optimizer_steps % 100 == 0:
             lrs_info_list = []
             if self.param_groups: # Use the property
                 for i,g_ in enumerate(self.param_groups):
                     lrs_info_list.append(f"G{i}_{g_.get('group_type', 'unk')}:{g_['lr']:.2e}")
             LOGGER.info(f"CustomPeftFairseqOptimizer.set_lr @ step {current_optimizer_steps}: LRs: {', '.join(lrs_info_list)}")

    def zero_grad(self, set_to_none=True):
        if self._optimizer: 
            self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None, groups=None):
        if self._optimizer:
            self._optimizer.step(closure)
    
    # state_dict and load_state_dict are inherited from FairseqOptimizer
    # and will delegate to self.optimizer (which is self._optimizer via property)

    @property
    def supports_flat_params(self):
        # If the wrapped torch optimizer (e.g. AdamW) doesn't support flat params itself,
        # or if we are not using flattened parameters for the FP32 master weights.
        # The FP16Optimizer.build_optimizer checks this for its fp32_optimizer.
        # If we use flatten=False when building fp32_master_params, this should be False.
        return False


def custom_trainer(Trainer_class): # Your main patching function name
    patch_flag_name = "_pefts_fp16_aware_patch_v2" # Consistent patch flag name
    if getattr(Trainer_class, patch_flag_name, False):
        LOGGER.debug(f"Fairseq Trainer already patched with {patch_flag_name}.")
        return
    
    LOGGER.info(f"Applying {patch_flag_name} to Fairseq Trainer...")
    setattr(Trainer_class, patch_flag_name, True) # Set the flag using setattr
    
    # Store original for fallbacks if needed, ensure this name is unique if multiple patches are ever combined
    Trainer_class._orig_build_optimizer_for_peft_patch = Trainer_class._build_optimizer 

    def _build_custom_peft_optimizer(self): # 'self' is the Trainer instance
        LOGGER.info(f"Custom patched _build_optimizer ({patch_flag_name}) CALLED.")

        # --- 1. Collect All Trainable Model and Criterion Parameters ---
        all_known_trainable_params_named = []
        param_to_full_name_map = {} 

        for name, p in self.model.named_parameters():
            if p.requires_grad:
                all_known_trainable_params_named.append(p) # Store param objects
                param_to_full_name_map[id(p)] = f"model.{name}"
        
        criterion_params_trainable_objects = []
        if utils.has_parameters(self.criterion):
            for name, p in self.criterion.named_parameters():
                if p.requires_grad:
                    criterion_params_trainable_objects.append(p) 
                    param_to_full_name_map[id(p)] = f"criterion.{name}"
        
        # For tagging original model params before creating FP32 masters
        original_model_params_trainable_objects = [p for name,p in self.model.named_parameters() if p.requires_grad]

        if not original_model_params_trainable_objects and not criterion_params_trainable_objects:
            LOGGER.error("CRITICAL: No trainable parameters found! Fallback to original optimizer build.");
            self._orig_build_optimizer_for_peft_patch(); return 
        LOGGER.info(f"Collected {len(original_model_params_trainable_objects)} trainable model params and {len(criterion_params_trainable_objects)} trainable criterion params.")

        lora_name_pattern = "lora_" 
        vera_trainable_name_pattern = "vera_lambda_"
        for p_orig in original_model_params_trainable_objects:
            name_orig = param_to_full_name_map.get(id(p_orig), "")
            is_lora = lora_name_pattern in name_orig
            is_vera = vera_trainable_name_pattern in name_orig
            if is_lora: p_orig._peft_group_type_tag = 'lora'
            elif is_vera: p_orig._peft_group_type_tag = 'vera'
            else: p_orig._peft_group_type_tag = 'base'

        # --- 2. Handle FP16 or FP32 parameter setup for the core optimizer ---
        fp16_enabled = getattr(self.cfg.common, 'fp16', False)
        params_for_core_torch_optimizer = [] 
        fp32_master_model_params_list = None 

        if fp16_enabled:
            if original_model_params_trainable_objects:
                LOGGER.info("FP16 enabled: Building FP32 master parameters for model (non-flattened).")
                fp32_master_model_params_list = FP16Optimizer.build_fp32_params(
                    self.cfg, # Pass main cfg, build_fp32_params uses cfg.optimizer
                    original_model_params_trainable_objects, 
                    flatten=False 
                )
                for i, p32_master in enumerate(fp32_master_model_params_list):
                    p32_master._peft_group_type_tag = getattr(original_model_params_trainable_objects[i], '_peft_group_type_tag', 'base')
                params_for_core_torch_optimizer.extend(fp32_master_model_params_list)
            if criterion_params_trainable_objects:
                for p_crit in criterion_params_trainable_objects: p_crit._peft_group_type_tag = 'base'
                params_for_core_torch_optimizer.extend(criterion_params_trainable_objects)
        else: 
            LOGGER.info("FP32 training: Using original model & criterion parameters directly.")
            for p_orig in original_model_params_trainable_objects: # Ensure tags are set
                 if not hasattr(p_orig, '_peft_group_type_tag'): p_orig._peft_group_type_tag = 'base' # Default if not PEFT
            for p_crit in criterion_params_trainable_objects: p_crit._peft_group_type_tag = 'base'
            params_for_core_torch_optimizer.extend(original_model_params_trainable_objects)
            params_for_core_torch_optimizer.extend(criterion_params_trainable_objects)

        # --- 3. Separate Core Optimizer Parameters (FP32) into PEFT groups ---
        base_core_params, lora_core_params, vera_core_params = [], [], []
        for p_core in params_for_core_torch_optimizer:
            group_tag = getattr(p_core, '_peft_group_type_tag', 'base') 
            if group_tag == 'lora': lora_core_params.append(p_core)
            elif group_tag == 'vera': vera_core_params.append(p_core)
            else: base_core_params.append(p_core)
        LOGGER.info(f"Separated core (FP32) params into: {len(base_core_params)} base, {len(lora_core_params)} LoRA, {len(vera_core_params)} VeRA.")

        # --- 4. Determine Initial Learning Rates for these groups ---
        try: initial_base_lr = self.cfg.optimizer.lr[0]
        except: initial_base_lr = 1e-4; LOGGER.error("Failed: get initial_base_lr.", exc_info=True)
        initial_vera_lr_cli = getattr(self.cfg.task, 'vera_lr', 0.0)
        initial_vera_lr = initial_vera_lr_cli if initial_vera_lr_cli > 0.0 else initial_base_lr
        initial_lora_lr_cli = getattr(self.cfg.task, 'lora_lr', 0.0) 
        initial_lora_lr = initial_lora_lr_cli if initial_lora_lr_cli > 0.0 else initial_base_lr
        LOGGER.info(f"Initial LRs for groups -- Base: {initial_base_lr:.2e}, LoRA: {initial_lora_lr:.2e}, VeRA: {initial_vera_lr:.2e}")

        # --- 5. Create Parameter Group Dictionaries for the Core (FP32) PyTorch Optimizer ---
        param_groups_for_core_optimizer = []
        if base_core_params: param_groups_for_core_optimizer.append({"params": base_core_params, "lr": initial_base_lr, "group_type": "base"})
        if lora_core_params: param_groups_for_core_optimizer.append({"params": lora_core_params, "lr": initial_lora_lr, "group_type": "lora"})
        if vera_core_params: param_groups_for_core_optimizer.append({"params": vera_core_params, "lr": initial_vera_lr, "group_type": "vera"})

        if not param_groups_for_core_optimizer: LOGGER.error("CRITICAL: No param groups for core optimizer."); self._orig_build_optimizer_for_peft_patch(); return

        # --- 6. Build the Core PyTorch Optimizer (operates on FP32 params) ---
        optimizer_name_str = self.cfg.optimizer._name 
        optimizer_cls = OPTIMIZER_REGISTRY.get(optimizer_name_str)
        if optimizer_cls is None: raise ValueError(f"Unsupported optimizer: {optimizer_name_str}")

        optimizer_kwargs = {"params": param_groups_for_core_optimizer}
        if optimizer_name_str == "adam":
            adam_betas_cfg = self.cfg.optimizer.adam_betas
            try: adam_betas_eval = ast.literal_eval(adam_betas_cfg) if isinstance(adam_betas_cfg, str) else adam_betas_cfg
            except: adam_betas_eval = (0.9,0.999); LOGGER.warning(f"Bad adam_betas: {adam_betas_cfg}. Defaulted.")
            optimizer_kwargs["betas"] = adam_betas_eval
            optimizer_kwargs["eps"] = self.cfg.optimizer.adam_eps
            optimizer_kwargs["weight_decay"] = self.cfg.optimizer.weight_decay
        else: 
            if hasattr(self.cfg.optimizer, 'weight_decay'): optimizer_kwargs["weight_decay"] = self.cfg.optimizer.weight_decay
            LOGGER.warning(f"Optimizer {optimizer_name_str} minimal setup.")
        
        torch_fp32_optimizer_instance = optimizer_cls(**optimizer_kwargs)
        LOGGER.info(f"Built core PyTorch FP32 optimizer: {type(torch_fp32_optimizer_instance).__name__}")
            
        # --- 7. Wrap the Core FP32 Optimizer with CustomPeftFairseqOptimizer ---
        fp32_fairseq_optimizer_with_custom_lr = CustomPeftFairseqOptimizer(self.cfg.optimizer, torch_fp32_optimizer_instance)
        
        if base_core_params: fp32_fairseq_optimizer_with_custom_lr.initial_lr_for_base_group = initial_base_lr
        if lora_core_params: 
            fp32_fairseq_optimizer_with_custom_lr.initial_lr_for_lora_group = initial_lora_lr
            fp32_fairseq_optimizer_with_custom_lr.lora_lr_schedule_type = getattr(self.cfg.task, 'lora_lr_schedule_type', 'proportional')
        if vera_core_params: 
            fp32_fairseq_optimizer_with_custom_lr.initial_lr_for_vera_group = initial_vera_lr
            fp32_fairseq_optimizer_with_custom_lr.vera_lr_schedule_type = getattr(self.cfg.task, 'vera_lr_schedule_type', 'proportional')
        LOGGER.info(f"Wrapped core FP32 optimizer. Schedule types - LoRA: {fp32_fairseq_optimizer_with_custom_lr.lora_lr_schedule_type}, VeRA: {fp32_fairseq_optimizer_with_custom_lr.vera_lr_schedule_type}")

        # --- 8. Conditionally Wrap with Fairseq's FP16Optimizer ---
        if fp16_enabled:
            LOGGER.info("FP16 enabled. Instantiating Fairseq's FP16Optimizer.")
            # FP16Optimizer signature: __init__(self, cfg: DictConfig, params, fp32_optimizer, fp32_params, **kwargs)
            # params: The original FP16 model parameters
            # fp32_optimizer: The FairseqOptimizer instance that works on FP32 master weights (our custom_fp32_fairseq_optimizer)
            # fp32_params: The list of FP32 master nn.Parameter objects for the model
            
            # Ensure original_model_params_trainable_objects and fp32_master_model_params_list are not None
            # if original_model_params_trainable_objects was populated.
            # fp32_master_model_params_list would be None if original_model_params_trainable_objects was empty.
            # However, FP16Optimizer.build_fp32_params would return [] if input params list is empty.

            if not original_model_params_trainable_objects and fp32_master_model_params_list:
                LOGGER.warning("FP16Optimizer: fp32_master_model_params_list exists but original_model_params_trainable_objects is empty. This is unusual.")
            if not fp32_master_model_params_list and original_model_params_trainable_objects:
                 LOGGER.warning("FP16Optimizer: original_model_params_trainable_objects exists but fp32_master_model_params_list is empty/None. This means build_fp32_params might have failed or was not called appropriately for model params.")
                 # If original_model_params_trainable_objects is not empty, fp32_master_model_params_list should also not be empty (it would be a list of FP32 params)
                 # For safety, if fp32_master_model_params_list is None but we expected it:
                 if original_model_params_trainable_objects and fp32_master_model_params_list is None:
                     LOGGER.error("CRITICAL: fp32_master_model_params_list is None when it should contain FP32 master params for FP16Optimizer. Falling back to non-FP16 path.")
                     self._optimizer = fp32_fairseq_optimizer_with_custom_lr # Fallback to custom FP32 optimizer
                 else:
                    # This path is taken if fp32_master_model_params_list is defined (even if empty)
                    self._optimizer = FP16Optimizer(
                        self.cfg,
                        original_model_params_trainable_objects, # <<<< CORRECTED VARIABLE NAME
                        fp32_fairseq_optimizer_with_custom_lr, 
                        fp32_master_model_params_list    
                    )
            elif not original_model_params_trainable_objects and not criterion_params_trainable_objects:
                # This case should be caught earlier, but as a safeguard for FP16Optimizer call
                LOGGER.warning("FP16Optimizer: No trainable model or criterion parameters to optimize with FP16. Using the custom FP32 optimizer as is (which might be empty).")
                self._optimizer = fp32_fairseq_optimizer_with_custom_lr
            else: # This is the main path if original_model_params_trainable_objects is not empty
                 self._optimizer = FP16Optimizer(
                    self.cfg,
                    original_model_params_trainable_objects, # <<<< CORRECTED VARIABLE NAME
                    fp32_fairseq_optimizer_with_custom_lr, 
                    fp32_master_model_params_list    
                )

            LOGGER.info(f"Wrapped with FP16Optimizer. Final Trainer optimizer: {type(self._optimizer).__name__}")
            if hasattr(self._optimizer, 'scaler') and self._optimizer.scaler is not None:
                 LOGGER.info(f"  FP16Optimizer scaler type: {type(self._optimizer.scaler).__name__}")
            else:
                 # This could be normal if cfg.common.bf16 is true, but we are only handling fp16 here.
                 # Or if scaler is explicitly disabled via config for FP16Optimizer.
                 LOGGER.warning("  FP16Optimizer does not have an active scaler. Check FP16 config if scaling is expected.")
        else: # FP32 training
            LOGGER.info("FP32 training. Using CustomPeftFairseqOptimizer directly for Trainer.")
            self._optimizer = fp32_fairseq_optimizer_with_custom_lr


        # --- 9. Build the Learning Rate Scheduler ---
        if not isinstance(self._optimizer, FairseqOptimizer):
            LOGGER.error(f"CRITICAL: self._optimizer not FairseqOptimizer. Type: {type(self._optimizer)}."); raise TypeError("Optimizer must be FairseqOptimizer.")
            
        if hasattr(self.cfg, 'lr_scheduler'):
            LOGGER.info(f"Building LR scheduler with: {getattr(self.cfg.lr_scheduler, '_name', 'N/A')}")
            self._lr_scheduler = lr_scheduler.build_lr_scheduler(self.cfg.lr_scheduler, self._optimizer) 
        else:
            LOGGER.error("CRITICAL: cfg.lr_scheduler not available!"); self._lr_scheduler = None

        if self._lr_scheduler is None:
            LOGGER.error("CRITICAL: Failed to build LR scheduler.")
            if getattr(self.cfg.lr_scheduler,'_name',None) not in [None,'none','fixed_without_scheduler']: # 'fixed_without_scheduler' is hypothetical
                raise RuntimeError(f"Failed to build LR scheduler '{self.cfg.lr_scheduler._name}'.")
        else:
            LOGGER.info(f"Built LR scheduler: {type(self._lr_scheduler).__name__} for final optimizer: {type(self._optimizer).__name__}")

    # In custom_trainer function:
    Trainer_class._build_optimizer = _build_custom_peft_optimizer
    LOGGER.info("Fairseq Trainer _build_optimizer patched with _build_custom_peft_optimizer (FP16-aware v2).")