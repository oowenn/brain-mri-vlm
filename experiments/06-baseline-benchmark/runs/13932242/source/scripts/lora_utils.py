"""Small, testable contracts for response-only LoRA training and selection."""
import math
import random


def supervised_sequences(prefixes, targets, pad_id):
    if not prefixes or len(prefixes) != len(targets) or any(not x for x in prefixes + targets):
        raise ValueError('Every example requires a prompt and a nonempty completion')
    width = max(len(p) + len(t) for p, t in zip(prefixes, targets))
    ids, masks, labels = [], [], []
    for prefix, target in zip(prefixes, targets):
        padding = width - len(prefix) - len(target)
        ids.append(prefix + target + [pad_id] * padding)
        masks.append([1] * (len(prefix) + len(target)) + [0] * padding)
        labels.append([-100] * len(prefix) + target + [-100] * padding)
    return ids, masks, labels


def epoch_order(size, seed, epoch):
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order


def choose_checkpoint(records):
    if not records or any(r['metrics']['n'] != 579
                          or not math.isfinite(r['metrics']['macro_f1'])
                          or not math.isfinite(r['validation_loss']) for r in records):
        raise ValueError('Checkpoint selection needs complete, finite validation results')
    return min(records, key=lambda r: (-r['metrics']['macro_f1'], r['validation_loss'], r['epoch']))


def merge_passes(config, report):
    return (report['probe_count'] == 4 * config['probe_per_class']
            and report['probe_token_matches'] == report['probe_count']
            and report['max_absolute_logit_difference'] <= config['max_absolute_logit_difference']
            and report['mean_absolute_logit_difference'] <= config['max_mean_absolute_logit_difference']
            and report['full_validation_label_agreement'] >= config['min_full_validation_label_agreement']
            and report['absolute_macro_f1_difference'] <= config['max_absolute_macro_f1_difference']
            and report['all_parameters_fp16'] and report['remaining_lora_modules'] == 0)
