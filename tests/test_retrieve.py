import pytest

from retrieve import RRF_K, reciprocal_rank_fusion, retrieve, tokenize


def test_tokenize_lowercases_and_keeps_decimal_numbers():
    assert tokenize("Best F1 was 0.7204, with BoW + LR!") == ["best", "f1", "was", "0.7204", "with", "bow", "lr"]


def test_tokenize_splits_slashes_so_each_word_can_match():
    assert tokenize("train/validation/test split") == ["train", "validation", "test", "split"]


def test_chunk_ranked_high_by_both_lists_comes_first():
    assert reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])[:2] in ([1, 2], [2, 1])
    assert reciprocal_rank_fusion([[5, 2, 3], [2, 6, 7]])[0] == 2


def test_chunk_found_by_only_one_list_is_still_kept():
    fused = reciprocal_rank_fusion([[1, 2], [3]])
    assert set(fused) == {1, 2, 3}


def test_fusion_uses_reciprocal_rank_scores():
    fused = reciprocal_rank_fusion([[10, 20], [20, 30]])
    scores = {10: 1 / (RRF_K + 1), 20: 1 / (RRF_K + 2) + 1 / (RRF_K + 1), 30: 1 / (RRF_K + 2)}
    assert fused == sorted(scores, key=lambda c: -scores[c])


def test_ties_are_broken_by_chunk_id_so_results_are_reproducible():
    assert reciprocal_rank_fusion([[7], [3]]) == [3, 7]


def test_empty_rankings_give_no_results():
    assert reciprocal_rank_fusion([[], []]) == []


def test_unknown_mode_is_rejected_before_loading_anything():
    with pytest.raises(ValueError):
        retrieve("question", mode="keyword")
