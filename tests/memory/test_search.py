"""Searching memories: which are returned for a query, and in what order.

What can go wrong: a word in another form not found, a script other than Latin
unsearchable, a memory matching more of the query ranked below one matching less, a
cutoff that drops a high score or decides a tie by rounding.
"""

from datetime import UTC, datetime

import pytest

from domyn.agents.memory.managers import FilesystemMemoryManager, Memory
from domyn.agents.memory.search import MemorySearch

SEARCH = MemorySearch()


def memories(*texts: str) -> list[Memory]:
    return [
        Memory(id=str(i), timestamp=datetime.now(UTC), memory=text) for i, text in enumerate(texts)
    ]


def found(entries: list[Memory], query: str, search: MemorySearch = SEARCH) -> list[str]:
    return [m.memory for m in search.search(entries, query)]


class TestWords:
    def test_case_and_accents_are_folded_every_script_kept(self):
        assert MemorySearch.words("Café À LA Crème, Пользователь Ёлка") == [
            "cafe",
            "a",
            "la",
            "creme",
            "пользователь",
            "елка",
        ]

    def test_an_underscore_splits_since_it_marks_the_edges(self):
        assert MemorySearch.words("snake_case") == ["snake", "case"]

    def test_trigrams_mark_where_a_word_starts_and_ends(self):
        assert MemorySearch.trigrams("italy") == ["_it", "ita", "tal", "aly", "ly_"]
        assert MemorySearch.trigrams("a") == ["_a_"]


class TestFinding:
    def test_a_word_in_another_form_is_found_first(self):
        entries = memories("The user lives in Italy", "The user prefers metric units")
        assert found(entries, "Italian food")[0] == "The user lives in Italy"
        assert found(entries, "preferences")[0] == "The user prefers metric units"

    def test_other_scripts_and_accents_are_searchable(self):
        entries = memories(
            "Пользователь любит чай", "Il codice della porta è 1234", "Je bois du café"
        )
        assert found(entries, "чай") == ["Пользователь любит чай"]
        assert found(entries, "è 1234")[0] == "Il codice della porta è 1234"
        assert found(entries, "cafe") == ["Je bois du café"]

    def test_a_memory_sharing_no_trigram_is_not_returned(self):
        entries = memories("The user drinks tea", "Пользователь любит чай")
        assert found(entries, "tea") == ["The user drinks tea"]

    def test_a_query_without_words_leaves_the_order_as_it_was(self):
        entries = memories("one", "two")
        assert found(entries, "?!") == ["one", "two"]

    def test_nothing_to_search_finds_nothing(self):
        assert SEARCH.search([], "tea") == []


class TestRanking:
    def test_a_memory_matching_more_of_the_query_comes_first(self):
        entries = memories(
            "The front door is green",
            "The garage code is 1234",
            "The front door code is 4321",
            "The user drinks tea",
        )
        assert found(entries, "front door code")[0] == "The front door code is 4321"

    def test_a_memory_with_both_words_comes_before_one_with_either(self):
        entries = memories(
            "The user likes coffee",
            "The user likes coffee and jazz",
            "The user likes jazz",
            "The user drinks tea",
        )
        assert found(entries, "coffee jazz")[0] == "The user likes coffee and jazz"

    def test_ties_keep_the_order_given(self):
        entries = memories("The user drinks tea", "The user drinks tea", "Something else")
        assert [m.id for m in SEARCH.search(entries, "tea")] == ["0", "1"]


class TestTheCutoff:
    def test_a_score_far_below_the_mean_is_dropped_high_ones_kept(self):
        # mean 7.25, deviation 3.63: the floor is 3.62.
        assert SEARCH.above_cutoff({0: 10.0, 1: 9.0, 2: 9.0, 3: 1.0}) == [0, 1, 2]

    def test_a_high_outlier_is_never_dropped(self):
        assert 0 in SEARCH.above_cutoff({0: 100.0, 1: 1.0, 2: 1.0, 3: 1.0})

    def test_with_one_or_two_scores_every_one_stays(self):
        # With two, the lower lies exactly one deviation below the mean: rounding
        # alone must not decide it.
        assert SEARCH.above_cutoff({0: 5.74, 1: 1.79}) == [0, 1]
        assert SEARCH.above_cutoff({0: 8.97, 1: 1.91}) == [0, 1]
        assert SEARCH.above_cutoff({0: 3.0}) == [0]

    def test_zero_keeps_only_what_is_at_or_above_the_mean(self):
        assert MemorySearch(cutoff_std=0).above_cutoff({0: 9.74, 1: 2.99, 2: 2.15}) == [0]

    def test_it_applies_to_a_search(self):
        entries = memories("The user works on a running app", "The user drinks tea every morning")
        strict = MemorySearch(cutoff_std=0)
        assert found(entries, "running", strict) == ["The user works on a running app"]


class TestSettings:
    def test_a_manager_is_configured_from_a_plain_dict(self, tmp_path):
        manager = FilesystemMemoryManager.from_config(
            {
                "type": "filesystem",
                "file_path": str(tmp_path / "m.md"),
                "search": {"cutoff_std": 0.5},
            }
        )
        assert manager.search.cutoff_std == 0.5

    def test_an_unknown_or_impossible_setting_is_refused(self):
        with pytest.raises(ValueError):
            MemorySearch(cutoff_std=-1)
        with pytest.raises(ValueError):
            MemorySearch(similarity=0.5)
