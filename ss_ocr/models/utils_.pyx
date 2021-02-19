# -*- coding: utf-8 -*-

# special compile-time information about the numpy module
cimport numpy as np

import numpy as np

################################################################################

def beam_search_decoder(np.ndarray data, int n_beams,
                        int top_paths=1, stop_tok=None):
    cdef list sequences = [(list(), np.float32(0.0), False)]
    cdef list all_candidates = []

    for row in data:
        all_candidates = []
        for seq, score, done in sequences:
            if not done:
                for i, log_prob in enumerate(row):
                    candidate = (seq + [i],
                                 score - log_prob,
                                 i == stop_tok)
                    all_candidates.append(candidate)
            else:
                all_candidates.append((seq, score, done))

        all_candidates.sort(key=lambda x: x[1])
        sequences = all_candidates[:n_beams]

    sequences = sequences[:top_paths]
    sequences = [(seq, score) for seq, score, _ in sequences]
    return sequences
