"""Offline regression checks for comparison parsing and evidence selection."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import hw4


class RetrievalRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_env = patch.dict(os.environ, {"HW4_VARIANT": "final"})
        cls.temp_env.start()
        cls.addClassCleanup(cls.temp_env.stop)
        data = Path(__file__).resolve().parent / "public_test"
        cls.questions = hw4.load_json(data / "questions.json")
        cls.gold = hw4.load_json(data / "ground_truth.json")
        cls.chunks = [
            c for name, html in hw4.load_corpus(data / "corpus").items()
            for c in hw4.chunk_document(name, html)
        ]
        cls.index = hw4.build_index(cls.chunks)

    def test_original_questions_parse_without_llm(self):
        with patch.object(hw4, "make_llm", side_effect=AssertionError("unexpected call")) as llm:
            for qid, question in self.questions.items():
                with self.subTest(qid=qid):
                    self.assertEqual(hw4.parse_score(hw4.parse_query(question), self.gold[qid]), 1)
            llm.assert_not_called()

    def test_rephrased_comparisons_keep_company_and_year_boundaries(self):
        base = self.questions["q031"]
        company_a, company_b = self.gold["q031"]["query"]["companies"]
        options = (
            f"Compare net income between {company_a} and {company_b} for FY2024.",
            f"Which company had higher net income: {company_a} or {company_b} in FY2024?",
            f"Compare net income: {company_a} vs. {company_b}, at the end of FY2024.",
            f"Between {company_a} and {company_b}, which had higher net income in FY2024?",
            f"Compare net income: {company_a.lower()} or {company_b.lower()} as of December 31, 2024.",
        )
        with patch.object(hw4, "make_llm", side_effect=AssertionError("unexpected call")) as llm:
            for question in (base, *options):
                with self.subTest(question=question):
                    self.assertEqual(hw4.parse_score(hw4.parse_query(question), self.gold["q031"]), 1)
            llm.assert_not_called()

    def test_lookalike_company_stays_out_of_corpus(self):
        question = "Compare revenue: Northgate Payments vs. Northgate Robotics Ventures for FY2024."
        parsed = hw4.parse_query(question)
        self.assertEqual(parsed["companies"], ["Northgate Payments", "Northgate Robotics Ventures"])
        with patch.object(hw4, "_llm_rerank_and_verify") as verify:
            self.assertEqual(hw4.retrieve(self.index, question), [])
            verify.assert_not_called()

    def test_keep_both_companies_when_reranker_selects_one(self):
        question = self.questions["q021"]
        expected = self.gold["q021"]["query"]["companies"]

        def rerank_one_company(question, candidates, check_answerable, is_comparison):
            self.assertTrue(is_comparison)
            lead = candidates[0].metadata["company"]
            return True, [i for i, chunk in enumerate(candidates, 1) if chunk.metadata["company"] == lead]

        with patch.object(hw4, "_llm_rerank_and_verify", side_effect=rerank_one_company):
            chosen = hw4.retrieve(self.index, question)
        self.assertLessEqual(len(chosen), hw4.MAX_RETRIEVED)
        self.assertEqual({c.metadata["company"] for c in chosen}, set(expected))
        self.assertTrue({id(c) for c in chosen} <= {id(c) for c in self.chunks})

    def test_second_pass_can_recover_or_confirm_rejection(self):
        question = self.questions["q004"]
        calls = []

        def recover(question, candidates, check_answerable, is_comparison):
            self.assertTrue(check_answerable and is_comparison)
            calls.append(candidates)
            return (False, []) if len(calls) == 1 else (True, [len(candidates)])

        with patch.object(hw4, "_llm_rerank_and_verify", side_effect=recover):
            chosen = hw4.retrieve(self.index, question)
        self.assertEqual(len(calls), 2)
        self.assertLess(len(calls[0]), len(calls[1]))
        self.assertIs(chosen[0], calls[1][-1])
        self.assertEqual(len({c.metadata["company"] for c in chosen}), 2)

        with patch.object(hw4, "_llm_rerank_and_verify", return_value=(False, [])) as verify:
            self.assertEqual(hw4.retrieve(self.index, question), [])
            self.assertEqual(verify.call_count, 2)

        with patch.object(hw4, "_llm_rerank_and_verify", side_effect=[(False, []), (None, [])]):
            self.assertTrue(hw4.retrieve(self.index, question))

    def test_malformed_verifier_output_is_not_a_rejection(self):
        class MalformedModel:
            def invoke(self, prompt):
                from types import SimpleNamespace

                return SimpleNamespace(content='{"answerable": "false", "order": [true, 1, 1, 900]}')

        with patch.object(hw4, "_RERANK_LLM", MalformedModel()):
            decision, order = hw4._llm_rerank_and_verify("test", self.chunks[:1], True, False)
        self.assertIsNone(decision)
        self.assertEqual(order, [1])


if __name__ == "__main__":
    unittest.main()
