"""Offline regression checks for omitted molecule judgment recovery."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

from domains.molecules.drd2.cards import MoleculeCandidateCard
from domains.molecules.drd2.run import judge_cohort, judge_with_coverage
from domains.molecules.drd2.teacher import MoleculeJudgmentResponse, build_judgment_request


def card(index):
    return MoleculeCandidateCard.model_validate({
        'representation_contract': 'molecule-candidate-card/1.0.0',
        'item_id': f'item-{index:03d}',
        'identity': {'status': 'observed'},
        'descriptors': {'status': 'observed'},
        'functional_groups': {'status': 'observed'},
        'geometry': {'status': 'observed'},
        'evidence_limitations': 'Diagnostic evidence only.',
    })


def response(*indices):
    return MoleculeJudgmentResponse.model_validate({'results': [{
        'item_id': f'item-{i:03d}', 'evidence_for': 'Observed evidence.',
        'evidence_against': 'Limited evidence.', 'critical_unknowns': 'Binding unknown.',
        'initial_label': 'abstain', 'self_critique': 'Evidence is limited.',
        'final_label': 'abstain', 'rationale': 'Binding is unknown.',
    } for i in indices]})


class JudgmentCoverageTests(unittest.TestCase):
    def setUp(self):
        self.cards = [card(i) for i in range(3)]
        self.request = build_judgment_request(
            self.cards, current_goal='Test goal', model='test', max_output_tokens=512,
            campaign_id='test', iteration=0, repeat_index=1)

    def execute(self, replies, limit=2):
        executor = Mock()
        executor.execute.side_effect = [(r, {'request': i}) for i, r in enumerate(replies)]
        result = judge_with_coverage(self.request, self.cards, executor=executor,
                                     maximum_redispatches=limit)
        return result, executor

    def test_complete_response_needs_no_recovery(self):
        (rows, metadata), executor = self.execute([response(2, 0, 1)])
        self.assertEqual([r.item_id for r in rows], [c.item_id for c in self.cards])
        self.assertEqual(executor.execute.call_count, 1)
        self.assertEqual(len(metadata), 1)

    def test_only_missing_items_are_resent_individually(self):
        (rows, metadata), executor = self.execute([response(0), response(1), response(2)])
        self.assertEqual([r.item_id for r in rows], [c.item_id for c in self.cards])
        self.assertEqual(len(metadata), 3)
        requests = [call.args[0] for call in executor.execute.call_args_list]
        self.assertEqual([[c['item_id'] for c in r.user_payload['candidates']]
                          for r in requests[1:]], [['item-001'], ['item-002']])
        self.assertEqual(len({r.request_hash for r in requests}), 3)
        (_, _), replay = self.execute([response(0), response(1), response(2)])
        self.assertEqual([c.args[0].request_hash for c in replay.execute.call_args_list],
                         [r.request_hash for r in requests])

    def test_duplicate_and_unknown_ids_fail_without_recovery(self):
        for reply in [response(0, 0), response(0, 9)]:
            executor = Mock()
            executor.execute.return_value = (reply, {})
            with self.subTest(reply=reply), self.assertRaisesRegex(ValueError, 'duplicate or unexpected'):
                judge_with_coverage(self.request, self.cards, executor=executor,
                                    maximum_redispatches=2)
            self.assertEqual(executor.execute.call_count, 1)

    def test_recovery_cannot_return_another_items_judgment(self):
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            self.execute([response(0, 2), response(0)])

    def test_retry_ceiling_stops_additional_calls(self):
        executor = Mock()
        executor.execute.side_effect = [(response(0), {}), (response(1), {})]
        with self.assertRaisesRegex(ValueError, 'retry limit exhausted.*item-002'):
            judge_with_coverage(self.request, self.cards, executor=executor,
                                maximum_redispatches=1)
        self.assertEqual(executor.execute.call_count, 2)

    def test_both_passes_recover_and_save_request_metadata(self):
        executor = Mock()
        executor.execute.side_effect = [
            (response(0, 2), {'request': 'first'}),
            (response(1), {'request': 'first-recovery'}),
            (response(0, 1), {'request': 'second'}),
            (response(2), {'request': 'second-recovery'}),
        ]
        config = {'experiment_id': 'test', 'candidate_judge': {
            'provider_order_seed': 1, 'chunk_size': 3, 'model': 'test',
            'max_output_tokens': 512, 'maximum_coverage_redispatches_per_chunk': 2,
        }}
        cohort = {'rows': [{'item_id': c.item_id, 'card': c.model_dump(mode='json')}
                           for c in self.cards],
                  'manifest': {'generation_attempt_count': 3}}
        with tempfile.TemporaryDirectory() as folder:
            result = judge_cohort(cohort, goal='Test goal', goal_id='goal', iteration=0,
                                  config=config, executor=executor, output_root=Path(folder))
            self.assertEqual(result['stable_labels'], {c.item_id: 'abstain' for c in self.cards})
            for repeat in (1, 2):
                record = json.loads((Path(folder) / 'goal' / f'repeat-{repeat}.json').read_text())
                self.assertEqual(len(record['requests']), 2)
            # Completed summaries replay without another provider call.
            judge_cohort(cohort, goal='Test goal', goal_id='goal', iteration=0,
                         config=config, executor=executor, output_root=Path(folder))
        self.assertEqual(executor.execute.call_count, 4)

    def test_provider_budget_failure_propagates(self):
        executor = Mock()
        executor.execute.side_effect = [(response(0, 2), {}), PermissionError('budget exhausted')]
        with self.assertRaisesRegex(PermissionError, 'budget exhausted'):
            judge_with_coverage(self.request, self.cards, executor=executor,
                                maximum_redispatches=2)


if __name__ == '__main__':
    unittest.main()
