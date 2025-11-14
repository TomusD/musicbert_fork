"""Implements multi-task token classification model.
"""

from itertools import count
import json
import logging
import math
import os
import pickle
import warnings
from typing import Sequence

import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import (
    Dictionary,
    FairseqDataset,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    RightPadDataset,
    SortDataset,
    data_utils,
)
from fairseq.tasks import FairseqTask, register_task
from torch import nn

import musicbert.monkeypatch_load_checkpoint
import musicbert.state_dict_patch
import musicbert.PeFTs.MHA_forward_patch as MHA_forward_patch
import musicbert.PeFTs.optimizer_patch as optimizer_patch
from musicbert.token_classification import RobertaSequenceTaggingHead
from musicbert.PeFTs.PeFTs import inject_lora, inject_vera
from musicbert.PeFTs.svft_layers import LinearWithSVFT, get_target_modules_list, create_and_replace_modules
from musicbert.PeFTs.MuMoE import inject_mumoe

LOGGER = logging.getLogger(__name__)

PAD_IDX = 1

# To get around the fact that some methods are static, we use a global dictionary
#   to store some attributes. Obviously this is a bit of a hack.

TARGET_INFO = {}


class AssertSameLengthDataset(FairseqDataset):
    def __init__(self, first, seconds, first_to_second_ratio: int = 1):
        self.first = first
        self.seconds = seconds
        self.first_to_second_ratio = first_to_second_ratio

    def __getitem__(self, index):
        for second in self.seconds:
            assert (
                torch.numel(self.first[index])
                == torch.numel(second[index]) * self.first_to_second_ratio
            )

    def __len__(self):
        return 0

    def collater(self, samples):
        return 0


class RobertaSequenceMultiTaggingHead(nn.Module):
    """Head for sequence tagging/token-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes: Sequence[int],
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
        liebel_loss: bool = False,
    ):
        super().__init__()
        sub_heads = []
        for n_class in num_classes:
            sub_heads.append(
                RobertaSequenceTaggingHead(
                    input_dim,
                    inner_dim,
                    n_class,
                    activation_fn,
                    pooler_dropout,
                    q_noise,
                    qn_block_size,
                    do_spectral_norm,
                )
            )
        self.n_heads = len(sub_heads)
        self.multi_tag_sub_heads = nn.ModuleList(sub_heads)
        if liebel_loss:
            # (Malcolm 2024-04-01) We actually use the loss_sigma parameter in the
            #   forward method of MultiTaskSequenceTaggingCriterion. This design
            #   seems like it could be improved upon to achieve better encapsulation.
            self.loss_sigma = nn.Parameter(
                torch.full((len(sub_heads),), 1 / len(sub_heads))
            )

    def forward(self, features, **kwargs):
        x = [sub_head(features) for sub_head in self.multi_tag_sub_heads]
        return x


@register_criterion("multitask_sequence_tagging")
class MultiTaskSequenceTaggingCriterion(FairseqCriterion):
    def __init__(self, task, classification_head_name):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.pad_idx = task.label_dictionaries[0].pad()
        self.compound_token_ratio = self.task.args.compound_token_ratio
        self.remaining_inputs_to_save = max(
            0, self.task.args.example_network_inputs_to_save
        )
        self.example_network_inputs_path = self.task.args.example_network_inputs_path
        self.target_dropout = self.task.args.target_dropout
        self.use_liebel_loss = self.task.args.liebel_loss

        # (Triantafulloy) ISNS weighting
        self.use_isns_weights = hasattr(self.task.args, 'use_isns_weights') and self.task.args.use_isns_weights
        if self.use_isns_weights:
            self.class_weights_list = self.task.class_weights_list
        else:
            self.class_weights_list = None

        if self.remaining_inputs_to_save:
            assert (
                self.example_network_inputs_path is not None
            ), "must provide --example-network-inputs-path if --example-network-inputs-to-save > 0"

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sequence_multitask_tagging_head',
                            help='name of the classification head to use')
        parser.add_argument('--compound-token-ratio', type=int, default=1)
        parser.add_argument('--example-network-inputs-to-save', type=int, default=0)
        parser.add_argument('--example-network-inputs-path', type=str, default=None)
        parser.add_argument('--target-dropout', type=float, default=0.0)
        parser.add_argument("--liebel-loss", action="store_true", help="use multi-task loss from Liebel and Korner 2018")
        parser.add_argument("--use-isns-weights",action="store_true",help="use ISNS weighting for the loss function",)
        # fmt: on

    def save_inputs(self, sample):
        def _save(key, tensor_or_dict):
            if isinstance(tensor_or_dict, dict):
                for sub_key, sub_val in tensor_or_dict.items():
                    _save(f"{key}_{sub_key}", sub_val)
            elif isinstance(tensor_or_dict, torch.Tensor):
                path = os.path.join(folder, f"{key}.npy")
                np.save(path, tensor_or_dict.detach().cpu().numpy())
            else:
                path = os.path.join(folder, f"{key}.pickle")
                with open(path, "wb") as outf:
                    pickle.dump(tensor_or_dict, outf)

        folder = os.path.join(
            self.example_network_inputs_path, f"{self.remaining_inputs_to_save}"
        )
        os.makedirs(folder, exist_ok=True)
        for key, tensor in sample.items():
            _save(key, tensor)
        self.remaining_inputs_to_save -= 1

    def get_logits(self, model, sample):
        multi_logits, _ = model(
            **sample["net_input"],
            features_only=True,
            return_all_hiddens=False,
            classification_head_name=self.classification_head_name,
        )
        return multi_logits

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sequence_tagging"

        if self.remaining_inputs_to_save:
            self.save_inputs(sample)

        multi_logits = self.get_logits(model, sample)

        adjusted_ntokens = sample["ntokens"] // self.compound_token_ratio
        nsentences = sample["target0"].size(0)
        sample_size = adjusted_ntokens - nsentences  # number of tokens without eos

        logging_output = {
            "ntokens": adjusted_ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            # because `reduce_metrics` is a static method we need to
            #   include the following in the logging output. (NB: if we make it
            #  nonstatic we will get an exception because fairseq calls it on the
            #   class rather than on an instance)
            "nspecial": self.task.label_dictionaries[0].nspecial,
        }
        losses = []
        masked_preds_list = []
        masked_targets_list = []
        for i, logits in enumerate(multi_logits):
            targets = sample[f"target{i}"].view(-1)
            logits = logits.view(-1, logits.size(-1))

            # (Triantafulloy) ISNS weighting
            current_weights = None
            if self.use_isns_weights and self.class_weights_list:
                current_weights = self.class_weights_list[i].to(logits.device)

            this_loss = F.nll_loss(
                F.log_softmax(logits, dim=-1, dtype=torch.float32),
                targets,
                ignore_index=self.pad_idx,
                reduction="sum",
                weight=current_weights
            )
            losses.append(this_loss)

            # To get the same behavior as the original implementation we should ignore
            #   all specials, not just pad. Not sure if we want to do this.

            these_masked_preds = (
                (logits[targets != self.pad_idx].argmax(dim=1)).detach().cpu().numpy()
            )
            these_masked_targets = (
                targets[targets != self.pad_idx].detach().cpu().numpy()
            )
            masked_preds_list.append(these_masked_preds)
            masked_targets_list.append(these_masked_targets)

            logging_output[f"loss_{i}"] = this_loss.data
            logging_output[f"y_true_{i}"] = these_masked_targets
            logging_output[f"y_pred_{i}"] = these_masked_preds
            logging_output[f"ncorrect_{i}"] = (
                these_masked_preds == these_masked_targets
            ).sum()

        masked_preds = np.concatenate(masked_preds_list)
        masked_targets = np.concatenate(masked_targets_list)

        if not model.training or not self.target_dropout:
            loss_stack = torch.stack(losses)
            if self.use_liebel_loss:
                loss_sigma = model.classification_heads[
                    self.classification_head_name
                ].loss_sigma

                # Note:
                # We use the loss function given in Qiu, Chen, and Zhang, “A Novel
                # Multi-Task Learning Method for Symbolic Music Emotion Recognition”:
                #
                # \sum{t} \frac{1}{2 * \sigma_t} + \ln * ( 1 + \sigma_{t}^{2})
                #
                # In Liebel and Körner, “Auxiliary Tasks in Multi-Task Learning”,
                # sigma_t in the denominator is squared as well:
                #
                # \sum{t} \frac{1}{2 * \sigma_{t}^{2}} + \ln * ( 1 + \sigma_{t}^{2})

                scaled_losses = 1 / (2 * loss_sigma) * loss_stack + torch.log(
                    1 + loss_sigma**2
                )
                loss = scaled_losses.sum()
            else:
                loss = loss_stack.mean()
        else:
            if self.use_liebel_loss:
                raise NotImplementedError
            loss_tensor = torch.stack(losses)

            rand_sample = torch.rand_like(loss_tensor)
            loss_mask = rand_sample > self.target_dropout
            if not loss_mask.any():
                loss_mask[torch.randint(loss_mask.shape[0], (1,))] = True
            loss_subset = torch.masked_select(loss_tensor, loss_mask)
            loss = loss_subset.mean()

        # TODO: (Malcolm 2023-09-15) allow weighting loss?

        logging_output.update(
            {
                "loss": loss.data,
                # "ncorrect": utils.item((masked_preds == masked_targets).sum()),
                "ncorrect": (masked_preds == masked_targets).sum(),
                "y_true": masked_targets,
                "y_pred": masked_preds,
            }
        )
        if self.use_liebel_loss:
            for i, sigma in enumerate(
                model.classification_heads[self.classification_head_name].loss_sigma
            ):
                logging_output[f"sigma_{i}"] = sigma.item()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        n_targets = TARGET_INFO["n_targets"]
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        # We follow fairseq implementation elsewhere here. Notes
        #   1. we divide by log(2) to convert from nats to bits
        #   2. "loss" is divided by sample_size but "nll_loss" is divided by ntokens,
        #       I'm not sure why. Note that ntokens should be the number of tokens
        #       including <eos> and sample_size should be the number of tokens
        #       excluding <eos>
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy",
                100.0 * ncorrect / (sample_size * n_targets),
                nsentences,
                round=1,
            )

        for i in count():
            # log liebel-loss parameters if they exist
            if len(logging_outputs) > 0 and f"sigma_{i}" in logging_outputs[0]:
                metrics.log_scalar(f"sigma_{i}", logging_outputs[-1][f"sigma_{i}"])
            else:
                break

        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )
        if len(logging_outputs) > 0 and "y_pred" in logging_outputs[0]:
            y_pred = np.concatenate(
                tuple(log.get("y_pred") for log in logging_outputs if "y_pred" in log)
            )
            y_true = np.concatenate(
                tuple(log.get("y_true") for log in logging_outputs if "y_true" in log)
            )
            for average in ["micro", "macro", "weighted"]:
                (
                    precision,
                    recall,
                    f1,
                    support,
                ) = sklearn.metrics.precision_recall_fscore_support(
                    y_true, y_pred, average=average, zero_division=0.0  # type:ignore
                )
                metrics.log_scalar(f"precision_{average}", precision)  # type:ignore
                metrics.log_scalar(f"recall_{average}", recall)  # type:ignore
                metrics.log_scalar(f"f1_{average}", f1)  # type:ignore
            balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
            metrics.log_scalar(f"balanced_accuracy", balanced_accuracy)

        n_special = TARGET_INFO["n_specials"]

        for target_i in range(n_targets):
            if not (
                len(logging_outputs) > 0 and f"y_pred_{target_i}" in logging_outputs[0]
            ):
                continue
            target_name = TARGET_INFO[f"target{target_i}_name"]

            ncorrect = sum(
                log.get(f"ncorrect_{target_i}", 0) for log in logging_outputs
            )
            metrics.log_scalar(
                f"accuracy_{target_name}",
                100.0 * ncorrect / sample_size,
                nsentences,
                round=1,
            )

            y_pred = np.concatenate(
                tuple(
                    log.get(f"y_pred_{target_i}")
                    for log in logging_outputs
                    if f"y_pred_{target_i}" in log
                )
            )
            y_true = np.concatenate(
                tuple(
                    log.get(f"y_true_{target_i}")
                    for log in logging_outputs
                    if f"y_true_{target_i}" in log
                )
            )

            for average in ["micro", "macro", "weighted"]:
                (
                    precision,
                    recall,
                    f1,
                    support,
                ) = sklearn.metrics.precision_recall_fscore_support(
                    y_true, y_pred, average=average, zero_division=0.0  # type:ignore
                )
                metrics.log_scalar(
                    f"precision_{target_name}_{average}", precision  # type:ignore
                )
                metrics.log_scalar(
                    f"recall_{target_name}_{average}", recall  # type:ignore
                )
                metrics.log_scalar(f"f1_{target_name}_{average}", f1)  # type:ignore
            balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
            metrics.log_scalar(f"balanced_accuracy", balanced_accuracy)

            if target_name not in TARGET_INFO["targets_to_log_by_label"]:
                continue
            # (Malcolm 2023-10-16) we could use
            #   sklearn.metrics.precision_recall_fscore_support with average=None to get
            #   the results per label instead
            no_specials_mask = y_true >= n_special  # type:ignore
            confused = sklearn.metrics.confusion_matrix(
                y_true[no_specials_mask] - n_special,
                y_pred[no_specials_mask] - n_special,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                precision_per_class = np.nan_to_num(
                    confused.diagonal() / confused.sum(axis=0)
                )
                recall_per_class = np.nan_to_num(
                    confused.diagonal() / confused.sum(axis=1)
                )
                f1_per_class = np.nan_to_num(
                    (2 * precision_per_class * recall_per_class)
                    / (precision_per_class + recall_per_class)
                )

            labels = TARGET_INFO[f"target{target_i}_vocab"]

            for (
                label_i,
                label,
            ) in enumerate(labels):
                # specials may or may not be included in the metric arrays, but we
                #   don't want to log them. So instead we do as follows:
                class_i = len(precision_per_class) - len(labels) + label_i
                metrics.log_scalar(
                    f"precision_{target_name}_{label}", precision_per_class[class_i]
                )
                metrics.log_scalar(
                    f"recall_{target_name}_{label}", recall_per_class[class_i]
                )
                metrics.log_scalar(f"f1_{target_name}_{label}", f1_per_class[class_i])

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_task("musicbert_multitask_sequence_tagging")
class MultiTaskSequenceTaggingTask(FairseqTask):
    """
    Sequence tagging (also called sentence tagging or sequence labelling) task that predicts a class for each input token.
    Inputs should be stored in 'input0' directory, labels in 'label' directory.
    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    upsample_encoder: bool = False

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        # -1 raises an exception below; --num-classes is required
        parser.add_argument(
            "--num-classes",
            nargs="+",
            type=int,
            default=-1,
            help="number of classes for each target (required)",
        )
        parser.add_argument("--ref-dir", default=None)
        parser.add_argument("--target-names", nargs="+", default=None)
        parser.add_argument("--msdebug", action="store_true")
        parser.add_argument("--freeze-layers", type=int, default=-1)
        parser.add_argument(
            "--targets-to-log-by-label",
            nargs="+",
            default=None,
            help="provide the name of targets for which we should log f1/etc. "
            "for each label individually",
        )
        # (Triantafyllou) PeFTs/MoE arguments
        # Composite learning rate had not straightforward implementation, thus not yet implemented.
        parser.add_argument( "--use-pefts", action="store_true", help="use patched forward for LoRA (default: False, use fairseq's default forward pass)")
        parser.add_argument("--use-custom-optimizer", action="store_true", help="Use patched custom optimizer to check if all PeFT parameters are being optimized (default: False, use fairseq's default optimizer)")
        parser.add_argument("--lora-rank",   type=int, default=0,help="LoRA rank (0 = no LoRA)")
        parser.add_argument("--lora-alpha",  type=int, default=1, help="LoRA scaling (alpha)")
        parser.add_argument("--lora-lr", type=float, default=0.0, help="LoRA learning rate (default: 0.0 = use base LR) not implemented yet")
        parser.add_argument("--lora-dropout",type=float, default=0.0, help="LoRA dropout probability")
        parser.add_argument("--rslora", action="store_true", help="use rsLoRA")
        parser.add_argument("--dora", action="store_true", help="use DoRA")
        parser.add_argument("--vera-rank", type=int, default=0, help="VeRA rank (0 = no VeRA)")
        parser.add_argument("--vera-lr", type=float, default=0.0, help="VeRA learning rate (default: 0.0 = use base LR) not implemented yet")
        parser.add_argument("--vera-dropout", type=float, default=0.0, help="VeRA dropout probability")
        parser.add_argument("--use-svft", action="store_true", help="use SVFT layers instead of linear layers (default: False)")
        parser.add_argument("--svft-off-diag", type=int, default=1, help="number of off-diagonals to use in SVFT layers for matrix M (default: 1)")
        parser.add_argument("--svft-pattern", type=str, default="banded", choices=["banded", "random", "top_k"], help="Sparsity pattern for SVFT's M matrix. (Using banded with off_diag=1 simulates SVFT-plain)")
        parser.add_argument("--svft-rank", type=int, default=768, help="SVFT rank (default full rank)")
        parser.add_argument("--use-mumoe", action="store_true", help="Replace unfrozen layers FFNs with MuMoE blocks")
        parser.add_argument("--mumoe-method", type=str, default="CP", choices=["CP", "TR"], help="Which MuMoE factorization method to use")
        parser.add_argument("--mumoe-layers", type=int, default=3, help="Number of layers to replace with MuMoE")
        parser.add_argument("--mumoe-num-experts", type=int, default=256, help="Number of experts for MuMoE")

    def __init__(self, args, data_dictionary, label_dictionaries):
        if args.msdebug:
            import pdb
            import sys
            import traceback

            def custom_excepthook(exc_type, exc_value, exc_traceback):
                traceback.print_exception(
                    exc_type, exc_value, exc_traceback, file=sys.stdout
                )
                pdb.post_mortem(exc_traceback)

            sys.excepthook = custom_excepthook
        super().__init__(args)
        MHA_forward_patch.set_use_pefts(args.use_pefts)
        optimizer_patch.set_use_custom_optimizer(args.use_custom_optimizer)
        self.dictionary = data_dictionary
        self._label_dictionaries = tuple(label_dictionaries)

        if not hasattr(args, "max_positions"):
            # TODO: (Malcolm 2023-09-08) the commented-out code will raise an attribute
            #   error. Instead we oblige the user to provide max positions as a CLI arg.
            # self._max_positions = (
            #     args.max_source_positions,
            #     args.max_target_positions,
            # )
            raise NotImplementedError("Provide '--max-positions'")
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions  # tuple[int, int] ?

        self.args = args

        # (Triantafulloy) Calculate class weights if ISNS weighting is enabled
        self.class_weights_list = []
        if self.args.use_isns_weights:
            for i, label_dict in enumerate(self._label_dictionaries):

                # Determine the number of output classes for this specific task's model head
                num_output_classes_for_task_head = self.args.num_classes[i] + label_dict.nspecial
                weights = torch.ones(num_output_classes_for_task_head, dtype=torch.float)

                for class_idx in range(num_output_classes_for_task_head):
                    if class_idx == label_dict.pad_index:
                        weights[class_idx] = 0.0
                    elif class_idx < len(label_dict.symbols):
                        count = label_dict.count[class_idx]
                        # ISNS: W = 100 * 1/sqrt(N_c). Treat count=0 as count=1 to avoid div by zero.
                        weights[class_idx] = 100.0 / torch.sqrt(torch.tensor(max(1, count), dtype=torch.float))
                    else:
                        weights[class_idx] = 1.0
                self.class_weights_list.append(weights)
                # --- DEBUG PRINT STATEMENT ---
                if self.args.use_isns_weights:
                    LOGGER.info(f"DEBUG: Task {i}:")
                    LOGGER.info(f"Number of output classes: {num_output_classes_for_task_head}")
                    LOGGER.info(f"Total {len(weights)} weights")
                    LOGGER.info(f"Calculated ISNS weights: {weights[:20]}")
                # --- END DEBUG ---

        self.num_targets = len(args.num_classes)
        if args.target_names is None:
            target_name_path = os.path.join(args.data, "target_names.json")
            assert os.path.exists(target_name_path)

            with open(target_name_path) as inf:
                self.target_names = json.load(inf)
        else:
            self.target_names = args.target_names

        assert self.num_targets == len(self.target_names)

        # Put necessary contents into TARGET_INFO
        n_specials_set = {d.nspecial for d in self._label_dictionaries}
        assert len(n_specials_set) == 1
        n_specials = tuple(n_specials_set)[0]
        TARGET_INFO["n_targets"] = self.num_targets
        TARGET_INFO["n_specials"] = n_specials
        if args.targets_to_log_by_label is not None:
            TARGET_INFO["targets_to_log_by_label"] = set(args.targets_to_log_by_label)
        else:
            TARGET_INFO["targets_to_log_by_label"] = set()
        for i, (target_name, n_classes, label_dictionary) in enumerate(
            zip(self.target_names, args.num_classes, self.label_dictionaries)
        ):
            targets = [label_dictionary[n_specials + j] for j in range(n_classes)]
            TARGET_INFO[f"target{i}_vocab"] = targets
            TARGET_INFO[f"target{i}_name"] = target_name

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        # (Malcolm 2023-09-05) We need the <mask> symbol not because we use it but
        #   so that the dictionary sizes match with the pretrained checkpoints.
        dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def _setup_task_helper(cls, args, **kwargs):
        assert isinstance(args.num_classes, list) and all(
            x > 0 for x in args.num_classes
        )

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "input0", "dict.txt"),
            source=True,
        )
        LOGGER.info("[input] dictionary: {} types".format(len(data_dict)))
        label_dicts = []
        for i in range(len(args.num_classes)):
            # load label dictionary
            ref_dir = args.ref_dir if args.ref_dir is not None else args.data
            label_dict = cls.load_dictionary(
                args,
                os.path.join(ref_dir, f"label{i}", f"dict.txt"),
                source=False,
            )
            label_dicts.append(label_dict)
            LOGGER.info(f"[label] dictionary {i}: {len(label_dict)} types")
        return data_dict, label_dicts

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_dict, label_dicts = cls._setup_task_helper(args, **kwargs)

        return cls(args, data_dict, label_dicts)

    def _load_dataset_helper_get_path(self, type, split):
        return os.path.join(self.args.data, type, split)  # type:ignore

    def _load_dataset_helper_make_dataset(self, type, dictionary, split, combine):
        split_path = self._load_dataset_helper_get_path(type, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            dictionary,
            self.args.dataset_impl,  # type:ignore
            combine=combine,
        )

        return dataset

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        src_tokens = self._load_dataset_helper_make_dataset(
            "input0", self.source_dictionary, split, combine
        )
        assert src_tokens is not None, "could not find dataset: {}".format(
            self._load_dataset_helper_get_path("input0", split)
        )

        dataset = {}
        label_datasets = []
        for i, label_dictionary in enumerate(self.label_dictionaries):
            label_dataset = self._load_dataset_helper_make_dataset(
                f"label{i}", label_dictionary, split, combine
            )
            if label_dataset is None:
                expected_path = self._load_dataset_helper_get_path(f"label{i}", split)
                LOGGER.warning(
                    f"could not find dataset: {expected_path}. If predicting "
                    "unlabeled data, this is expected."
                )
                continue

            # (Malcolm 2023-09-08) The PR I initially based the token classification
            #   code off of includes the following commented out lines so that we only
            #   predict items in the target vocabulary and not specials. However that
            #   doesn't seem necessary and there seem to be some weird bugs related to
            #   it on so I'm disabling it for now at least.

            # OffsetTokensDataset offsets tokens to get the targets to the
            # correct range (0,1,2,...)
            # label_dataset = OffsetTokensDataset(
            #     label_dataset,
            #     offset=-self.label_dictionary.nspecial,
            # )
            # ReplaceDataset replaces specials (bos, eos, and existing padding used when some
            # tokens should not be predicted) with -1
            # label_dataset = ReplaceDataset(
            #     label_dataset,
            #     replace_map={i: -1 for i in range(-self.label_dictionary.nspecial, 0)},
            #     offsets=np.zeros(len(label_dataset), dtype=int),
            # )
            # RightPadDataset uses -1 as padding, will be used to mask out padding
            # when calculating loss
            label_dataset = RightPadDataset(
                label_dataset, pad_idx=label_dictionary.pad()
            )
            assert label_dictionary.pad() == self.source_dictionary.pad() == PAD_IDX

            dataset[f"target{i}"] = label_dataset
            label_datasets.append(label_dataset)

        dataset.update(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": RightPadDataset(
                        src_tokens,
                        pad_idx=self.source_dictionary.pad(),
                    ),
                    "src_lengths": NumelDataset(src_tokens, reduce=False),
                },
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(src_tokens, reduce=True),
                "_assert_lengths_match": AssertSameLengthDataset(
                    src_tokens, label_datasets, self.args.compound_token_ratio
                ),
            }
        )

        nested_dataset = NestedDictionaryDataset(dataset, sizes=[src_tokens.sizes])

        with data_utils.numpy_seed(self.args.seed):  # type:ignore
            shuffle = np.random.permutation(len(src_tokens))
        dataset = SortDataset(nested_dataset, sort_order=[shuffle])

        LOGGER.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def _build_model_freeze_helper(self, args, model):
        if args.freeze_layers > 0:
            LOGGER.info(f"Freezing {args.freeze_layers=} layers")
            # What we *don't* want to freeze:
            # 1. the last n - freeze_layers encoder layers
            # 2. the classification head
            n_layers = len(model.encoder.sentence_encoder.layers)
            assert n_layers >= args.freeze_layers

            for parameter in model.parameters():
                parameter.requires_grad = False

            for layer_i, layer in enumerate(model.encoder.sentence_encoder.layers):
                if layer_i < args.freeze_layers:
                    continue
                for parameter in layer.parameters():
                    parameter.requires_grad = True

            if model.encoder.sentence_encoder.upsample:
                for parameter in model.encoder.sentence_encoder.upsampling.parameters():
                    parameter.requires_grad = True

    def _build_model_num_classes_helper(self):
        num_classes = []
        for n, dict in zip(self.args.num_classes, self.label_dictionaries):
            num_classes.append(n + dict.nspecial)
        return num_classes

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)
        self._build_model_freeze_helper(args, model)

        # (Triantafulloy) Inject LoRA adapters
        if args.lora_rank > 0 and not args.vera_rank:
            lora_target_modules=["q_proj","k_proj","v_proj","out_proj", "fc1","fc2"]
            replacements_made = inject_lora(
                module=model.encoder.sentence_encoder,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                rslora=args.rslora,
                dora=args.dora,
                target_modules=lora_target_modules
            )

            LOGGER.info(f"Made {replacements_made} LoRA replacements.")

            # Unfreeze LoRA parameters so the optimizer can find them
            lora_params_unfrozen = 0
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                    lora_params_unfrozen += 1
            LOGGER.info(f"Unfroze {lora_params_unfrozen} LoRA parameters.")

            # Debugging which LoRA architecture is being used and how many trainable parameters
            if args.rslora:
                LOGGER.info("Using rsLoRA")
            if args.dora:
                LOGGER.info("Using DoRA")
            if args.lora_rank > 0 and not args.rslora and not args.dora:
                LOGGER.info("Using Vanilla LoRA ") 

            # Debugging LoRA/DoRA layers
            all_params = [n for n, _ in model.named_parameters()]
            LOGGER.info(f"All parameters: {len(all_params)} ")

            lora_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
            LOGGER.info(f"LoRA/DoRA trainable parameters: {len(lora_param_names)} ")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    LOGGER.info(f"{name} - {param.shape}")

            frozen_lora_param_names = [n for n, p in model.named_parameters() if not p.requires_grad]
            LOGGER.info(f"LoRA/DoRA frozen parameters: {len(frozen_lora_param_names)} ")
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    LOGGER.info(f"{name} - {param.shape}")

        # (Triantafulloy) Inject VeRA adapters
        if args.vera_rank > 0 and not args.lora_rank:
            vera_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
            LOGGER.info("Using VeRA")

            # VeRA needs 2 frozen matrices A and B so I initialize them here
            hidden_size = model.encoder.sentence_encoder.layers[0].self_attn.embed_dim

            # Initialize the VeRA tensors A and B
            vera_A_tensor = torch.empty(args.vera_rank, hidden_size)
            vera_B_tensor = torch.empty(hidden_size, args.vera_rank)

            # Use Kaiming uniform initialization for VeRA matrices since worked best in the paper
            torch.nn.init.kaiming_uniform_(vera_A_tensor, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(vera_B_tensor, a=math.sqrt(5))

            # Create the VeRA frozen matrices
            vera_A_param_dict = nn.ParameterDict({
                "default": nn.Parameter(vera_A_tensor, requires_grad=False)
            })
            vera_B_param_dict = nn.ParameterDict({
                "default": nn.Parameter(vera_B_tensor, requires_grad=False)
            })
            LOGGER.info(f"Created shared VeRA matrices within ModuleDict: A{tuple(vera_A_tensor.shape)}, B{tuple(vera_B_tensor.shape)}")

            replacements_made = inject_vera(
                module=model.encoder.sentence_encoder,
                vera_A=vera_A_param_dict,
                vera_B=vera_B_param_dict,
                rank=args.vera_rank,
                dropout=args.vera_dropout,
                target_modules=vera_target_modules
            )

            LOGGER.info(f"Made {replacements_made} VeRA replacements.")

            # Unfreeze only the trainable parts of VeRA
            vera_params_unfrozen = 0
            for name, param in model.named_parameters():
                if "vera_lambda_" in name:
                    param.requires_grad = True
                    vera_params_unfrozen += 1
            LOGGER.info(f"Unfroze {vera_params_unfrozen} VeRA trainable parameters.")

            # Debugging VeRA layers
            all_params = [n for n, _ in model.named_parameters()]
            LOGGER.info(f"All parameters: {len(all_params)} ")

            vera_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
            LOGGER.info(f"VeRA trainable parameters: {len(vera_param_names)} ")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    LOGGER.info(f"{name} - {param.shape}")

            frozen_vera_param_names = [n for n, p in model.named_parameters() if not p.requires_grad]
            LOGGER.info(f"VeRA frozen parameters: {len(frozen_vera_param_names)} ")
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    LOGGER.info(f"{name} - {param.shape}")

        # (Triantafulloy) Inject SVFT Layers
        if args.use_svft:
            svft_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            LOGGER.info(f"Applying SVFT with pattern='{args.svft_pattern}', rank={args.svft_rank}, off_diagonals={args.svft_off_diag}")
            
            # Use the helper function from svft.py to get the full paths to these modules
            target_modules_list = get_target_modules_list(model.encoder.sentence_encoder, svft_target_modules)
            LOGGER.info(f"Found {len(target_modules_list)} target modules to replace with SVFT layers.")

            # Define a function that creates a new SVFT layer
            def create_svft_fn(original_linear_layer):
                return LinearWithSVFT(
                    linear=original_linear_layer,
                    off_diag=args.svft_off_diag,
                    pattern=args.svft_pattern,
                    rank=args.svft_rank,
                    fill_orthonormal=False
                )
            
            # Use the helper function from svft.py to perform the replacement
            create_and_replace_modules(model.encoder.sentence_encoder, target_modules_list, create_svft_fn)

            # Debugging SVFT layers
            all_params = [n for n, _ in model.named_parameters()]
            LOGGER.info(f"All parameters: {len(all_params)} ")

            svft_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
            LOGGER.info(f"SVFT trainable parameters: {len(svft_param_names)} ")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    LOGGER.info(f"{name} - {param.shape}")

            frozen_svft_param_names = [n for n, p in model.named_parameters() if not p.requires_grad]
            LOGGER.info(f"SVFT frozen parameters: {len(frozen_svft_param_names)} ")
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    LOGGER.info(f"{name} - {param.shape}")

        # (Triantafulloy) Inject MuMoE layers
        if args.use_mumoe:
            total_layers = len(model.encoder.sentence_encoder.layers)
            unfrozen_layers = total_layers - args.freeze_layers
            num_to_replace = min(args.mumoe_layers, unfrozen_layers)
            
            if num_to_replace <= 0:
                raise ValueError(f"No unfrozen layers available to replace.")

            target_indices = list(range(total_layers - num_to_replace, total_layers))

            mumoe_params = {
                'moe_method': args.mumoe_method,
                'num_experts': args.mumoe_num_experts
            }
            inject_mumoe(model, target_indices, mumoe_params)
            
            # Debugging MuMoE layers
            all_params = [n for n, _ in model.named_parameters()]
            LOGGER.info(f"All parameters: {len(all_params)} ")

            mumoe_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
            LOGGER.info(f"MuMoE trainable parameters: {len(mumoe_param_names)} ")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    LOGGER.info(f"{name} - {param.shape}")

            frozen_mumoe_param_names = [n for n, p in model.named_parameters() if not p.requires_grad]
            LOGGER.info(f"MuMoE frozen parameters: {len(frozen_mumoe_param_names)} ")
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    LOGGER.info(f"{name} - {param.shape}")
            

        # Logging trainable parameters after all PeFTs modifications
        final_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
        LOGGER.info(f"Final trainable parameters in the model: {len(final_param_names)} ")
        for name, param in model.named_parameters():
            if param.requires_grad:
                LOGGER.info(f"{name} - {param.shape} - Trainable")

        num_classes = self._build_model_num_classes_helper()

        # We register the sequence tagging head after any freezing so that it won't
        #   be frozen

        model.register_multitask_sequence_tagging_head(
            getattr(args, "classification_head_name", "sequence_multitask_tagging_head"),
            num_classes=num_classes,
            sequence_tagging=True,
        )

        # # Composite optimizer groups for base and PeFTs parameters
        # LOGGER.info("Assigning parameters to optimizer groups ('base' and 'vera')...")
        # for name, p in model.named_parameters():
        #     if not p.requires_grad:
        #         continue
            
        #     # Assign trainable parameters to the each group
        #     if "vera_" in name:
        #         p.param_group = "vera"
        #     elif "lora_" in name:
        #         p.param_group = "lora"
        #     else:
        #         p.param_group = "base"

        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionaries(self):
        return self._label_dictionaries
