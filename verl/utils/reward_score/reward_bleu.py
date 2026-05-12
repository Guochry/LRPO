import os
import re
import csv
import torch
from sacrebleu.metrics import BLEU
from threading import Lock


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    bleu = BLEU(tokenize="zh", effective_order=True, smooth_method="exp")
    reward = bleu.sentence_score(solution_str, [ground_truth]).score / 100.0

    return reward
