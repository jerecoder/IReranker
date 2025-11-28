from __future__ import annotations

from .oracle import MatrixOracle


class UnidirectionalMatrixOracle(MatrixOracle):
    """
    Oracle backed by rerank matrices that store (qid, doc_a, doc_b).

    sample_lt(i, j) returns True if the matrix indicates that doc_i
    (doc_a) is preferred over doc_j (doc_b), based only on the
    (qid, doc_i, doc_j) direction.
    """

    def __init__(self, base_dir=None):
        super().__init__(base_dir=base_dir, cache_comparisons=True)

    def sample_lt(self, i: int, j: int) -> bool:
        matrix = self._ensure_matrix_loaded()

        if self.current_task is None:
            return False

        qid = self.current_task.query_id
        doc_a = self.current_task.candidate_ids[i]
        doc_b = self.current_task.candidate_ids[j]

        entry = matrix.get((qid, doc_a, doc_b))
        if entry is None:
            return False

        pref = self._entry_preference(entry)
        if pref is None:
            return False

        return pref == "A"
