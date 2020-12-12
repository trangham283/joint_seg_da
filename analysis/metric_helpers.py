import code
import json
from collections import Counter

import numpy as np
import jiwer
from sklearn.metrics import f1_score

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for (i, c1) in enumerate(s1):
        current_row = [i + 1]
        for (j, c2) in enumerate(s2):
            insertions = previous_row[j + 1] + 1  # j+1 instead of j since
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def instance_metrics(ref_labels, hyp_labels):
    segment_records = []
    n_segment_tokens, n_segment_seg_errors, n_segment_joint_errors = 0, 0, 0
    for ref, hyp in zip(ref_labels, hyp_labels):
        n_segment_tokens += 1
        if hyp[0] != ref[0]:
            n_segment_seg_errors += 1
        if hyp != ref:
            n_segment_joint_errors += 1
        if ref.startswith("E"):
            segment_records.append((n_segment_tokens, n_segment_seg_errors, n_segment_joint_errors))
            n_segment_tokens, n_segment_seg_errors, n_segment_joint_errors = 0, 0, 0
    
    n_segments = len(segment_records)
    n_tokens = 0
    n_wrong_seg_segments = 0
    n_wrong_seg_tokens = 0
    n_wrong_joint_segments = 0
    n_wrong_joint_tokens = 0
    for (n_segment_tokens, n_segment_seg_errors, n_segment_joint_errors) in segment_records:
        n_tokens += n_segment_tokens
        if n_segment_seg_errors > 0:
            n_wrong_seg_segments += 1
            n_wrong_seg_tokens += n_segment_tokens
        if n_segment_joint_errors > 0:
            n_wrong_joint_segments += 1
            n_wrong_joint_tokens += n_segment_tokens

    DSER = n_wrong_seg_segments / n_segments
    strict_seg_err = n_wrong_seg_tokens / n_tokens
    DER = n_wrong_joint_segments / n_segments
    strict_joint_err = n_wrong_joint_tokens / n_tokens

    ref_short = [x for x in ref_labels if x != "I"]
    hyp_short = [x for x in hyp_labels if x != "I"]
    lwer = jiwer.wer(ref_short, hyp_short)
    return {
        "DSER": DSER,
        "strict segmentation error": strict_seg_err,
        "DER": DER,
        "strict joint error": strict_joint_err,
        "LWER": lwer
    }

def batch_metrics(refs, hyps):
    score_lists = {
        "DSER": [],
        "strict segmentation error": [],
        "DER": [],
        "strict joint error": [],
        "LWER": []
    }
    for ref_labels, hyp_labels in zip(refs, hyps):
        this_metrics = instance_metrics(ref_labels, hyp_labels)
        for k, v in this_metrics.items():
            score_lists[k].append(v)

    flattened_refs = [label for ref in refs for label in ref]
    flattened_hyps = [label for hyp in hyps for label in hyp]
    macro_f1 = f1_score(flattened_refs, flattened_hyps, average="macro")
    micro_f1 = f1_score(flattened_refs, flattened_hyps, average="micro")
    flat_ref_short = [x for x in flattened_refs if x != "I"]
    flat_hyp_short = [x for x in flattened_hyps if x != "I"]
    lwer = jiwer.wer(flat_ref_short, flat_hyp_short)

    return {
        "DSER": np.mean(score_lists["DSER"]),
        "strict segmentation error": np.mean(score_lists["strict segmentation error"]),
        "DER": np.mean(score_lists["DER"]),
        "strict joint error": np.mean(score_lists["strict joint error"]),
        "Macro F1": macro_f1,
        "Micro F1": micro_f1,
        "Macro LWER": np.mean(score_lists["LWER"]),
        "Micro LWER": lwer,
    }

def instance_metrics_asr(ref_labels, hyp_labels, dist_fn=levenshtein):
    ref_short = [x for x in ref_labels if x != "I"]
    hyp_short = [x for x in hyp_labels if x != "I"]
    lwer = jiwer.wer(ref_short, hyp_short)

    ler = jiwer.wer(ref_labels, hyp_labels)
    
    t_ids = [i for i, t in enumerate(ref_labels) if "E" in t]
    r_ids = [i for i, r in enumerate(hyp_labels) if "E" in r]
    s = 0
    for t in t_ids: s += min([abs(r - t) for r in r_ids])
    for r in r_ids: s += min([abs(r - t) for t in t_ids])
        
    ser = s / 2 / len(ref_short)
    nser = abs(len(ref_short) - len(hyp_short)) / len(ref_short)
    
    new_ref = []
    new_hyp = []
    offset = 0
    for i in t_ids:
        new_ref += [ref_labels[i]] * (i - offset + 1)
        offset = i+1 
    offset = 0
    for i in r_ids:
        new_hyp += [hyp_labels[i]] * (i - offset + 1)
        offset = i+1 
    daer = jiwer.wer(new_ref, new_hyp)
    return {"LWER": lwer,
            "LER": ler,
            "SER": ser,
            "NSER": nser,
            "DAER": daer}

def batch_metrics_asr(refs, hyps, dist_fn=levenshtein):
    score_lists = {
        "LWER": [],
        "LER": [],
        "SER": [],
        "NSER": [],
        "DAER": []
    }
    for ref_labels, hyp_labels in zip(refs, hyps):
        this_metrics = instance_metrics_asr(ref_labels, hyp_labels)
        for k, v in this_metrics.items():
            score_lists[k].append(v)

    flattened_refs = [label for ref in refs for label in ref]
    flattened_hyps = [label for hyp in hyps for label in hyp]
    flat_ref_short = [x for x in flattened_refs if x != "I"]
    flat_hyp_short = [x for x in flattened_hyps if x != "I"]
    lwer = jiwer.wer(flat_ref_short, flat_hyp_short)
    ler = jiwer.wer(flattened_refs, flattened_hyps)
    
    t_ids = [i for i, t in enumerate(flattened_refs) if "E" in t]
    r_ids = [i for i, r in enumerate(flattened_hyps) if "E" in r]
    s = 0
    for t in t_ids: s += min([abs(r - t) for r in r_ids])
    for r in r_ids: s += min([abs(r - t) for t in t_ids])
    ser = s / 2 / len(t_ids)
    
    nser = abs(len(t_ids) - len(r_ids)) / len(t_ids)
    
    new_ref = []
    new_hyp = []
    offset = 0
    for i in t_ids:
        new_ref += [flattened_refs[i]] * (i - offset + 1)
        offset = i+1 
    offset = 0
    for i in r_ids:
        new_hyp += [flattened_hyps[i]] * (i - offset + 1)
        offset = i+1 
    daer = jiwer.wer(new_ref, new_hyp)

    return {
        "Macro LWER": np.mean(score_lists["LWER"]),
        "Micro LWER": lwer,
        "Macro LER": np.mean(score_lists["LER"]),
        "Micro LER": ler,
        "Macro SER": np.mean(score_lists["SER"]),
        "Micro SER": ser,
        "Macro NSER": np.mean(score_lists["NSER"]),
        "Micro NSER": nser,
        "Macro DAER": np.mean(score_lists["DAER"]),
        "Micro DAER": daer,
    }


