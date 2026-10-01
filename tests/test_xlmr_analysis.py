"""Verifica a análise e a arquitetura com dados sintéticos, sem baixar pesos."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from transformers import XLMRobertaConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import epitome_xlmr as e


def synthetic_pairs():
    rows = []
    for prompt in range(20):
        for model in e.MODELS:
            strong_ex = (prompt < 13 if model == 'GEMMA' else
                         prompt in set(range(11)) | {12, 13} if model == 'LLAMA' else prompt in {12, 13})
            rows.append({'prompt_id': prompt, 'id': model, 'ER_label': int(model == 'LLAMA' and prompt < 2),
                         'IP_label': 0, 'EX_label': 2 * int(strong_ex)})
    return pd.DataFrame(rows)


class AnalysisTests(unittest.TestCase):
    def test_exact_and_holm_against_independent_implementations(self):
        _, tests = e.analysis_tables(synthetic_pairs())
        expected = []
        for r in tests.itertuples():
            n = r.A_score_higher + r.B_score_higher
            expected.append(binomtest(r.A_score_higher, n).pvalue if n else 1.)
        np.testing.assert_allclose(tests.p_exact_paired_sign, expected)
        np.testing.assert_allclose(tests.p_holm_9, multipletests(expected, method='holm')[1])
        ex = tests[(tests.metric == 'EX') & (tests.model_B == 'GPT4')]
        np.testing.assert_allclose(ex.p_holm_9, [.02734375, .0087890625])

    def test_quasi_separation_without_optimizer_warning(self):
        regression, status = e.regression_table(synthetic_pairs())
        self.assertFalse(status['ER']['available'])
        self.assertEqual(set(status['ER']['constant_outcome_models']), {'GEMMA', 'GPT4'})
        self.assertFalse(status['IP']['available'])
        self.assertTrue(status['EX']['available'])
        self.assertEqual(set(regression.metric), {'EX'})
        self.assertTrue(np.isfinite(regression.log_odds).all())
        gpt = regression[regression.model == 'GPT4'].iloc[0]
        self.assertAlmostEqual(gpt.odds_ratio, (2 / 18) / (13 / 7), places=8)

    def test_architecture_and_ignored_rationales(self):
        torch.set_num_threads(1)
        config = XLMRobertaConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
                                 pad_token_id=1, bos_token_id=0, eos_token_id=2)
        model = e.EpitomeXLMR(config=config)
        ids = torch.tensor([[0, 4, 5, 2, 1], [0, 6, 7, 2, 1]])
        mask = (ids != 1).long()
        result = model(ids, mask, ids, mask, labels=torch.tensor([0, 2]), rationale_labels=torch.full_like(ids, -100))
        self.assertEqual(tuple(result['logits'].shape), (2, 3))
        self.assertEqual(tuple(result['rationale_logits'].shape), (2, 5, 2))
        self.assertTrue(torch.isfinite(result['loss']))
        result['loss'].backward()
        self.assertTrue(all(p.grad is None for p in model.seeker.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.responder.parameters()))


if __name__ == '__main__':
    unittest.main()
