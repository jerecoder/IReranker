from __future__ import annotations

from ireranker.types import RankingTask

from .oracle import MatrixOracle


class BidirectionalMatrixOracle(MatrixOracle):
    """Oracle backed by rerank matrices that store (qid, doc_a, doc_b) and its reverse."""

    def sample_lt(self, i: int, j: int) -> bool:
        matrix = self._ensure_matrix_loaded()

        if self.current_task is None:
            return False

        qid = self.current_task.query_id
        doc_a = self.current_task.candidate_ids[i]
        doc_b = self.current_task.candidate_ids[j]

        forward_key = (qid, doc_a, doc_b)
        reverse_key = (qid, doc_b, doc_a)

        forward_entry = matrix.get(forward_key)
        reverse_entry = matrix.get(reverse_key)
        if forward_entry is None or reverse_entry is None:
            return False

        forward_pref = self._entry_preference(forward_entry)
        reverse_pref = self._entry_preference(reverse_entry)
        if forward_pref is None or reverse_pref is None:
            return False

        return forward_pref == "B" and reverse_pref == "A"
