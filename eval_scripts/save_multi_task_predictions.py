"""
Example testing command:
python eval_scripts/save_multi_task_predictions.py --data-dir ~/project/datasets/chord_tones/fairseq/many_target_bin --checkpoint ~/project/new_checkpoints/musicbert_fork/32702693/checkpoint_best.pt --output-folder ~/tmp/mout --msdebug --ignore-specials 4 --overwrite --max-examples 2
python eval_scripts/save_multi_task_predictions.py \
    --data-dir ~/output/test_data/chord_tones_bin \
    --checkpoint ~/output/musicbert_checkpoints/32702693/checkpoint_best.pt \
    --output-folder ~/tmp/mout --msdebug --ignore-specials 4 \
    --overwrite --max-examples 2
"""

import argparse
import json
import logging
import os
import shutil
import sys
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fairseq.data.dictionary import Dictionary
from fairseq.models.roberta import RobertaModel
from fairseq.models.roberta.hub_interface import RobertaHubInterface

SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))
PARENT_DIR = os.path.join(SCRIPT_DIR, "..")

USER_DIR = os.path.join(SCRIPT_DIR, "..", "musicbert")
sys.path.append(PARENT_DIR)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


LOG_INTERVAL = 50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        required=True,
        help="assumed to end in '_bin' and have an equivalent ending in '_raw' that contains 'metadata_test.txt'",
    )
    parser.add_argument(
        "--ref-dir",
        default=None,
        help="a directory that contains `target_names.json` as well as "
        "`label[x]/dict.txt` files. If not provided, the value of "
        "--data-dir is used.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="test", choices=("test", "valid", "train"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--compound-token-ratio", type=int, default=8)
    parser.add_argument("--msdebug", action="store_true")
    parser.add_argument("--overwrite", "-o", action="store_true")
    parser.add_argument("--ignore-specials", type=int, default=4)
    parser.add_argument(
        "--task", default="musicbert_multitask_sequence_tagging", type=str
    )
    parser.add_argument("--head", default="sequence_multitask_tagging_head", type=str)
    # (Triantafyllou)
    parser.add_argument("--interpretation-data-path", type=str, default=None,
                        help="Save detailed coefficient and metadata to this path as a CSV.")
    parser.add_argument("--layer-coeffs", type=int, default=11,help="The layer index to extract MuMoE coefficients from. Default is 11, which is the last layer in the encoder.")
    # TODO: Not yet implemented
    #parser.add_argument("--zero-out-expert", type=int, default=None, help="If provided, all coefficients for this expert will be set to 0.")

    args = parser.parse_args()
    return args


def extract_features_with_conditioning(
    self,
    tokens: torch.LongTensor,
    z_tokens: torch.LongTensor,
    return_all_hiddens: bool = False,
) -> torch.Tensor:
    # Due to fairseq's somewhat weird import system overriding RobertaHubInterface
    #   is somewhat tricky, so instead we monkey-patch
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)  # type:ignore
    if tokens.size(-1) > self.model.max_positions():
        raise ValueError(
            "tokens exceeds maximum length: {} > {}".format(
                tokens.size(-1), self.model.max_positions()
            )
        )
    if z_tokens.dim() == 1:
        z_tokens.unsqueeze(0)

    features, extra = self.model(
        tokens.to(device=self.device),  # type:ignore
        features_only=True,
        return_all_hiddens=return_all_hiddens,
        z_tokens=z_tokens.to(device=self.device),  # type:ignore
    )
    if return_all_hiddens:
        # convert from T x B x C -> B x T x C
        inner_states = extra["inner_states"]
        return [
            inner_state.transpose(0, 1) for inner_state in inner_states
        ]  # type:ignore
    else:
        return features  # just the last layer's features


def predict_with_conditioning(
    self,
    head: str,
    tokens: torch.LongTensor,
    z_tokens: torch.LongTensor,
    return_logits: bool = False,
):
    features = self.extract_features(
        tokens.to(device=self.device), z_tokens=z_tokens.to(device=self.device)
    )  # type:ignore
    logits = self.model.classification_heads[head](features)
    if return_logits:
        return logits
    return F.log_softmax(logits, dim=-1)


def main():
    args = parse_args()

    data_dir, ref_dir, checkpoint, output_folder_base = (
        args.data_dir,
        args.ref_dir,
        args.checkpoint,
        args.output_folder,
    )
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

    if ref_dir is None:
        ref_dir = data_dir

    output_folder = os.path.join(output_folder_base, args.dataset)

    if os.path.exists(output_folder):
        if args.overwrite:
            shutil.rmtree(output_folder)
        else:
            raise ValueError(f"Output folder {output_folder} already exists")

    assert data_dir.rstrip(os.path.sep).endswith("_bin")

    with open(os.path.join(ref_dir, "target_names.json"), "r") as inf:
        target_names = json.load(inf)

    if args.task == "musicbert_conditioned_multitask_sequence_tagging":
        RobertaHubInterface.extract_features = (
            extract_features_with_conditioning
        )  # type:ignore
        RobertaHubInterface.predict = predict_with_conditioning  # type:ignore

    musicbert = RobertaModel.from_pretrained(
        model_name_or_path=PARENT_DIR,
        checkpoint_file=checkpoint,
        data_name_or_path=data_dir,
        user_dir=USER_DIR,
        task=args.task,
        ref_dir=args.ref_dir,
        target_names=target_names,
    )

    musicbert.task.load_dataset(args.dataset)
    dataset = musicbert.task.datasets[args.dataset]

    # (Triantafyllou) If interpretation data path is provided we need map sample IDs to score IDs
    metadata_basename = f"metadata_{args.dataset}.txt"
    metadata_path = os.path.join(data_dir, metadata_basename)

    # Clear metadata file to save only the relevant columns
    note_indices = {}
    try:
        metadata_df = pd.read_csv(metadata_path, usecols=['score_id', 'csv_path', 'start_offset', 'df_indices'])
        LOGGER.info(f"Loaded metadata for {len(metadata_df)} samples from {metadata_path}")
        for idx, row in metadata_df.iterrows():
            df_indices_str = row['df_indices']
            indices_str = df_indices_str.strip('[]')
            indices = [int(i.strip()) for i in indices_str.split(',')]
            note_indices[idx] = indices
        LOGGER.info(f"Extracted note indices for {len(note_indices)} samples")
    except FileNotFoundError:
        LOGGER.error(f"Metadata file {metadata_path} not found. Please check the path.")
        return

    if torch.cuda.is_available():
        musicbert.cuda()

    musicbert.eval()

    n_examples = len(dataset)
    if args.max_examples is not None:
        n_examples = min(args.max_examples, n_examples)

    os.makedirs(output_folder, exist_ok=False)
    os.makedirs(os.path.join(output_folder, "predictions"), exist_ok=False)

    outfs = {}
    out_hdfs: dict[str, h5py.File] = {}

    label_dictionaries: dict[str, Dictionary] = {}
    for i, target_name in enumerate(target_names):
        outfs[target_name] = open(
            os.path.join(output_folder, "predictions", f"{target_name}.txt"), "w"
        )
        dictionary = musicbert.task.label_dictionaries[i]
        label_dictionaries[target_name] = dictionary
        dictionary.save(os.path.join(output_folder, f"{target_name}_dictionary.txt"))
        out_hdfs[target_name] = h5py.File(
            os.path.join(output_folder, "predictions", f"{target_name}.h5"), "w"
        )

    interpret_score = []
    def decode_timesig(timesig_token):
        if isinstance(timesig_token, int) and 0 <= timesig_token < len(TS_LIST):
            return TS_LIST[timesig_token]
        else:
            LOGGER.warning(f"Invalid timesig token: {timesig_token} and type of: {type(timesig_token)}. Returning default 4/4")
            return (4, 4) 
        
    # Build time signature lookup table
    TS_LIST = []
    MAX_TS_DENOMINATOR = 6
    MAX_NOTES_PER_BAR = 2
    for i in range(0, MAX_TS_DENOMINATOR + 1):  # denominators up to 2^6 (64)
        for j in range(1, ((2**i) * MAX_NOTES_PER_BAR) + 1):  # various numerators
            TS_LIST.append((j, 2**i))

    try:
        for batch_i, i in enumerate(range(0, n_examples, args.batch_size)):
            # TODO: remove this after debugging
            #if batch_i < 155:
                #continue
            samples = [
                dataset[j] for j in range(i, min(n_examples, i + args.batch_size))
            ]
            batch = dataset.collater(samples)
            src_tokens = batch["net_input"]["src_tokens"]

            predict_kwargs = {
                # TODO rename
                "head": args.head,
                "tokens": src_tokens,
                "return_logits": True,
            }

            if "z_tokens" in batch:
                predict_kwargs["z_tokens"] = batch["z_tokens"]

            all_logits = musicbert.predict(  # type:ignore
                **predict_kwargs
            )

            # (Triantafyllou) Save MuMoE coefficients
            if args.interpretation_data_path:
                model = musicbert.model
                sample_ids_in_batch = batch["id"]
                source_dict = musicbert.task.source_dictionary
                num_special_symbols = source_dict.nspecial
                encoder_layers = model.encoder.sentence_encoder.layers
                mumoe_method = None
                n_experts = None


                for layer in encoder_layers:
                    if "MuMoETransformerLayer" in type(layer).__name__:
                        if "CPMuMoE" in type(layer.ffn_block.MuMoE).__name__:
                            mumoe_method = "CP"
                        else:
                            mumoe_method = "TR"
                    
                        n_experts = layer.ffn_block.a[0].shape[2]
                        break

                # The order of features in OctupleMIDI according to the MusicBERT paper
                octuple_feature_map = {
                    0: 'bar',
                    1: 'pos',
                    2: 'instrument',
                    3: 'pitch',
                    4: 'duration',
                    5: 'velocity',
                    6: 'timesig',
                    7: 'tempo'
                }
                num_tokens = 8

                # Layer to extract coefficients from
                layer = model.encoder.sentence_encoder.layers[args.layer_coeffs] 

                if "MuMoETransformerLayer" in type(layer).__name__ and hasattr(layer, 'ffn_block') and hasattr(layer.ffn_block, 'a'):
                    coeffs_tensor = layer.ffn_block.a[0]

                    # TODO: Not yet implemented
                    # # Zero out the specified expert if requested
                    # if args.zero_out_expert is not None:
                    #     if args.zero_out_expert < coeffs_tensor.size(2):
                    #         coeffs_tensor[:, :, args.zero_out_expert] = 0.0
                    #         LOGGER.info(f"Zeroed out coefficients for expert {args.zero_out_expert} in layer {args.layer_coeffs}")
                    #     else:
                    #         LOGGER.warning(f"Expert index {args.zero_out_expert} is out of range. Model has {coeffs_tensor.size(2)} experts.")
                
                    # Process each sample in the current batch
                    for batch_sample_idx in range(sample_ids_in_batch.size(0)):
                        sample_id = sample_ids_in_batch[batch_sample_idx].item()
                        events_for_sample = []
                        min_bar_sample = float('inf')
                        bar_to_timesig = {} 
                        seq_len_for_batch = src_tokens.size(1)

                        # First pass over tokens of current sample j to parse events and find its min_bar
                        for start_token_idx in range(0, seq_len_for_batch, num_tokens):                      
                            event_properties = {}
                            is_valid_musical_event_chunk = False

                            # Feature extraction for the current octuple
                            for token_in_event in range(num_tokens):
                                current_token_idx = start_token_idx + token_in_event
                                token_id = src_tokens[batch_sample_idx, current_token_idx].item()

                                if token_id < num_special_symbols:
                                    continue
                            
                                is_valid_musical_event_chunk = True
                                token_str = source_dict.symbols[token_id].strip('<>')
                            
                                try:
                                    feature_idx, feature_val = map(int, token_str.split('-'))
                                    if feature_idx in octuple_feature_map:
                                        feature_name = octuple_feature_map[feature_idx]
                                        event_properties[feature_name] = feature_val
                                        if feature_idx == 6 and 'bar' in event_properties:
                                            bar_to_timesig[event_properties['bar']] = feature_val
                                except (ValueError, IndexError):
                                    LOGGER.warning(f"Malformed token: {token_str} in sample {sample_id}, skipping")
                                    continue
                        
                            # Only store if it's an octuple that defines a pitch
                            if is_valid_musical_event_chunk and 'pitch' in event_properties:
                                events_for_sample.append({
                                    'properties': event_properties,
                                    'start_token_idx': start_token_idx
                                })
                                if 'bar' in event_properties:
                                    min_bar_sample = min(min_bar_sample, event_properties['bar'])

                        # If no 'bar' token was found in any event of this sample, default min_bar to 0
                        if min_bar_sample == float('inf'):
                            LOGGER.warning(f"No 'bar' token found in sample {sample_id}, defaulting min_bar_sample to 0")
                            min_bar_sample = 0
                    
                        # Second pass: Use collected events and min_bar_sample to create scores
                        for note_index, event_data in enumerate(events_for_sample):
                            properties = event_data['properties']
                            k = event_data['start_token_idx']

                            # Calculate relative bar and the new offset
                            bar_val = properties.get('bar')
                            relative_bar = bar_val - min_bar_sample

                            timesig_token = properties.get('timesig')
                            ts_num, ts_den = decode_timesig(timesig_token)
                        
                            # Calculate beats per bar based on time signature
                            beats_per_bar = (ts_num * 4) / ts_den 
                            offset_in_beats = (relative_bar * beats_per_bar) + (properties.get('pos') / 16.0)

                            #Calculate global offset using the start offset from metadata
                            start_offset = metadata_df.loc[sample_id, 'start_offset']
                            global_offset = start_offset + offset_in_beats
                            
                            # Calculate global measure
                            start_measures = start_offset // beats_per_bar
                            global_measure = relative_bar + start_measures

                            # Calculate tempo
                            tempo_token = properties.get('tempo')
                            tempo = 2 ** (tempo_token / 12) * 16

                            # Duration, Measure, Global Measure, Offset, Global Offset, Tempo, Time Signature can be dropped. 
                            # OctupleMidi encoding is lossy so they are not perfectly aligned with the original score.
                            # We keep them here for reference.
                            note = {
                                'sample_id': sample_id,
                                'token_index': k,
                                'note_index': note_indices[sample_id][note_index],
                                'score_id': metadata_df.loc[sample_id, 'score_id'],
                                'csv_path': metadata_df.loc[sample_id, 'csv_path'],
                                'pitch': properties.get('pitch'),
                                'duration': properties.get('duration'),
                                'measure': relative_bar,
                                'global_measure': global_measure,
                                'offset': offset_in_beats,
                                'global_offset': global_offset,
                                'tempo': tempo,
                                'ts_numerator': ts_num,
                                'ts_denominator': ts_den
                            }
                                                    
                            # Collect coefficients for this octuple
                            compound_idx = k // num_tokens
                            event_coeffs = coeffs_tensor[batch_sample_idx, compound_idx, :]

                            for expert_idx in range(coeffs_tensor.size(2)): # num_experts
                                note[f'coeff_expert_{expert_idx}'] = event_coeffs[expert_idx].item()

                            # Get predictions to map with tokens
                            for logits, target_name in zip(all_logits, target_names):
                                preds = logits.argmax(dim=-1).detach().cpu().numpy()
                                if batch_sample_idx < preds.shape[0] and note_index < preds.shape[1]:
                                    note[f'{target_name}_pred_class'] = int(preds[batch_sample_idx, note_index])
                                    note[f'{target_name}_pred_label'] = label_dictionaries[target_name].string([preds[batch_sample_idx, note_index]])
                        
                            interpret_score.append(note)

                    if args.interpretation_data_path and interpret_score:
                        LOGGER.info(f"Saving expert's coefficients data for {len(interpret_score)} notes in batch {batch_i}...")
                        batch_df = pd.DataFrame(interpret_score)
                        zero_expert_index = f"_zero_exp_{args.zero_out_expert}" if args.zero_out_expert is not None else ""
                        bach_path = os.path.join(os.path.dirname(args.interpretation_data_path), f"{mumoe_method}_interpretation_data_{n_experts}_layer_{args.layer_coeffs}{zero_expert_index}.csv")
                        os.makedirs(os.path.dirname(bach_path), exist_ok=True)
                        file_exists = os.path.isfile(bach_path)
                        batch_df.to_csv(bach_path, mode='a' if file_exists else 'w', header=not file_exists, index=False)
                        LOGGER.info(f"Saved {len(interpret_score)} notes to {bach_path}")
                        interpret_score = []                           

            for logits, target_name in zip(all_logits, target_names):
                # logits: batch x seq x vocab

                # Enumerate over batch dimension
                for logit_i, example in enumerate(logits, start=i):
                    # Trim start and end tokens:
                    data = example.detach().cpu().numpy()[1:-1]

                    # Don't save specials
                    if args.ignore_specials:
                        data = data[:, args.ignore_specials :]

                    out_hdfs[target_name].create_dataset(f"logits_{logit_i}", data=data)

                preds = logits.argmax(dim=-1)
                target_lengths = (
                    batch["net_input"]["src_lengths"] // args.compound_token_ratio
                )
                for line, n_tokens in zip(preds, target_lengths):
                    pred_tokens = label_dictionaries[target_name].string(
                        line[:n_tokens]
                    )
                    outfs[target_name].write(pred_tokens)
                    outfs[target_name].write("\n")
            if batch_i and (batch_i % LOG_INTERVAL == 0):
                LOGGER.info(f"Batch {batch_i}")
    finally:
        for outf in outfs.values():
            outf.close()

        for outf in out_hdfs.values():
            outf.close()

    metadata_basename = f"metadata_{args.dataset}.txt"
    metadata_path = os.path.join(data_dir, metadata_basename)
    if not os.path.exists(metadata_path):
        raw_data_dir = data_dir.rstrip(os.path.sep)[:-4] + "_raw"
        assert os.path.exists(raw_data_dir)
        metadata_path = os.path.join(raw_data_dir, metadata_basename)

    shutil.copy(metadata_path, os.path.join(output_folder, metadata_basename))

    if args.ignore_specials:
        with open(os.path.join(output_folder, "num_ignored_specials.txt"), "w") as outf:
            outf.write(str(args.ignore_specials))


if __name__ == "__main__":
    main()
