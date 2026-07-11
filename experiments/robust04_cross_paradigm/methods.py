from __future__ import annotations

import re

from ireranker.oracles import BudgetExceeded

from experiments.robust04_cross_paradigm.engine import SharedFlanT5Engine, UsageMeter


SETWISE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _complete_ranking(prefix: list[str], original: list[str]) -> list[str]:
    seen = set(prefix)
    return prefix + [doc_id for doc_id in original if doc_id not in seen]


def run_prp_heapsort(
    query: str,
    candidate_ids: list[str],
    documents: dict[str, str],
    *,
    engine: SharedFlanT5Engine,
    meter: UsageMeter,
    k: int = 10,
) -> list[str]:
    original = list(candidate_ids)
    order = list(candidate_ids)
    extracted: list[str] = []
    query = engine.truncate_query(query)
    text_cache = {doc_id: engine.truncate_passage(documents[doc_id]) for doc_id in order}

    def better(a: str, b: str) -> bool:
        return engine.compare_bidirectional(query, text_cache[a], text_cache[b], meter=meter)

    def heapify(n: int, i: int) -> None:
        largest = i
        left, right = 2 * i + 1, 2 * i + 2
        if left < n and better(order[left], order[i]):
            largest = left
        if right < n and better(order[right], order[largest]):
            largest = right
        if largest != i:
            order[i], order[largest] = order[largest], order[i]
            heapify(n, largest)

    try:
        n = len(order)
        for i in range(n // 2, -1, -1):
            heapify(n, i)
        for i in range(n - 1, 0, -1):
            order[i], order[0] = order[0], order[i]
            extracted.append(order[i])
            if len(extracted) >= k:
                break
            heapify(i, 0)
    except BudgetExceeded:
        pass
    return _complete_ranking(extracted, original)


def render_setwise(query: str, passages: list[str]) -> str:
    body = "\n\n".join(
        f'Passage {SETWISE_LABELS[index]}: "{text}"' for index, text in enumerate(passages)
    )
    return (
        f'Given a query "{query}", which of the following passages is the most relevant one '
        f'to the query?\n\n{body}\n\nOutput only the passage label of the most relevant passage:'
    )


def run_setwise_heapsort(
    query: str,
    candidate_ids: list[str],
    documents: dict[str, str],
    *,
    engine: SharedFlanT5Engine,
    meter: UsageMeter,
    num_child: int = 2,
    k: int = 10,
) -> list[str]:
    original = list(candidate_ids)
    order = list(candidate_ids)
    extracted: list[str] = []
    query = engine.truncate_query(query)
    text_cache = {doc_id: engine.truncate_passage(documents[doc_id]) for doc_id in order}

    def choose(indices: list[int]) -> int:
        prompt = render_setwise(query, [text_cache[order[index]] for index in indices])
        output = engine.generate(
            [prompt], meter=meter, max_new_tokens=2, decoder_prefix=True,
            document_counts=[len(indices)],
        )[0]
        meter.choice_events += 1
        normalized = str(output).strip().upper()
        label = normalized[-1:] if normalized else ""
        if label not in SETWISE_LABELS[: len(indices)]:
            meter.invalid_outputs += 1
            return indices[0]
        return indices[SETWISE_LABELS.index(label)]

    def heapify(n: int, i: int) -> None:
        first_child = num_child * i + 1
        if first_child >= n:
            return
        indices = [i] + list(range(first_child, min(num_child * (i + 1) + 1, n)))
        largest = choose(indices)
        if largest != i:
            order[i], order[largest] = order[largest], order[i]
            heapify(n, largest)

    try:
        n = len(order)
        for i in range(n // num_child, -1, -1):
            heapify(n, i)
        for i in range(n - 1, 0, -1):
            order[i], order[0] = order[0], order[i]
            extracted.append(order[i])
            if len(extracted) >= k:
                break
            heapify(i, 0)
    except BudgetExceeded:
        pass
    return _complete_ranking(extracted, original)


def render_listwise(query: str, passages: list[str]) -> str:
    num = len(passages)
    message = (
        "This is RankGPT, an intelligent assistant that can rank passages based on their "
        "relevancy to the query.\n\n"
        f"The following are {num} passages, each indicated by number identifier []. "
        f"I can rank them based on their relevance to query: {query}\n\n"
    )
    for rank, content in enumerate(passages, start=1):
        content = content.replace("Title: Content: ", "").strip()
        message += f"[{rank}] {content}\n\n"
    message += f"The search query is: {query}\n\n"
    message += (
        f"I will rank the {num} passages above based on their relevance to the search query. "
        "The passages will be listed in descending order using identifiers, and the most "
        "relevant passages should be listed first, and the output format should be [] > [] > "
        "etc, e.g., [1] > [2] > etc.\n\n"
        f"The ranking results of the {num} passages (only identifiers) is:"
    )
    return message


def _apply_permutation(window: list[str], output: str) -> list[str]:
    response = [int(value) - 1 for value in re.findall(r"\d+", str(output))]
    unique: list[int] = []
    for value in response:
        if 0 <= value < len(window) and value not in unique:
            unique.append(value)
    unique.extend(value for value in range(len(window)) if value not in unique)
    return [window[value] for value in unique]


def run_listwise_rankgpt(
    query: str,
    candidate_ids: list[str],
    documents: dict[str, str],
    *,
    engine: SharedFlanT5Engine,
    meter: UsageMeter,
    window_size: int = 4,
    step_size: int = 2,
    repeats: int = 5,
) -> list[str]:
    ranking = list(candidate_ids)
    query = engine.truncate_query(query)
    text_cache = {doc_id: engine.truncate_passage(documents[doc_id]) for doc_id in ranking}
    try:
        for _ in range(repeats):
            end = len(ranking)
            start = end - window_size
            while start >= 0:
                window = ranking[start:end]
                prompt = render_listwise(query, [text_cache[doc_id] for doc_id in window])
                output = engine.generate(
                    [prompt], meter=meter, max_new_tokens=20, decoder_prefix=False,
                    document_counts=[len(window)],
                )[0]
                meter.choice_events += 1
                valid_labels = [int(value) for value in re.findall(r"\d+", str(output))]
                if sorted(valid_labels) != list(range(1, len(window) + 1)):
                    meter.invalid_outputs += 1
                ranking[start:end] = _apply_permutation(window, output)
                end -= step_size
                start -= step_size
    except BudgetExceeded:
        pass
    return ranking
