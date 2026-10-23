import pytest

from cirrus.decoding.prompt_lookup import PromptLookupDrafter, accept_greedy, choose_draft


def test_drafts_continuation_of_repeated_ngram():
    d = PromptLookupDrafter(max_ngram=3, min_ngram=2, max_draft=4)
    # "source" contains 5 6 7 8 9 after 1 2; the answer later starts 1 2 ...
    d.extend([1, 2, 5, 6, 7, 8, 9, 0, 3, 1, 2])
    assert d.draft() == [5, 6, 7, 8]


def test_prefers_longest_ngram_and_most_recent_occurrence():
    d = PromptLookupDrafter(max_ngram=3, min_ngram=1, max_draft=2)
    d.extend([4, 2, 7, 7, 9, 2, 8, 8, 4, 2])
    # 3-gram (8,4,2) unseen, 2-gram (4,2) seen at start -> continues 7, 7
    assert d.draft() == [7, 7]
    d2 = PromptLookupDrafter(max_ngram=1, min_ngram=1, max_draft=1)
    d2.extend([2, 7, 2, 8, 2])
    assert d2.draft() == [8]  # most recent occurrence of 2 wins


def test_no_match_returns_empty():
    d = PromptLookupDrafter()
    d.extend([1, 2, 3, 4])
    assert d.draft() == []


def test_accept_greedy():
    assert accept_greedy([5, 6, 7], [5, 6, 9, 1]) == (2, [5, 6, 9])
    assert accept_greedy([5, 6], [5, 6, 3]) == (2, [5, 6, 3])
    assert accept_greedy([], [4]) == (0, [4])
    with pytest.raises(ValueError):
        accept_greedy([1], [1])


def test_choose_draft():
    assert choose_draft([1, 2, 3], [9, 9]) == [1, 2, 3]
    assert choose_draft([1], [9, 9]) == [9, 9]
    assert choose_draft([], []) == []
