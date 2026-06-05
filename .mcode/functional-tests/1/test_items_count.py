# -*- coding: utf-8 -*-
"""
Functional tests for ItemsCount utility.

Verifies:
- Percentage mode ("20%") returns correct slice
- Count mode ("5") returns correct slice
- Integer mode (3) returns correct slice
- Edge cases: 0%, very large count, single item
"""
import pytest
from sumy.utils import ItemsCount


SAMPLE = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class TestItemsCountPercentage:
    """ItemsCount with percentage string values."""

    def test_20_percent(self):
        count = ItemsCount("20%")
        result = count(SAMPLE)
        # 10 * 20 // 100 = 2
        assert len(result) == 2
        assert result == [0, 1]

    def test_50_percent(self):
        count = ItemsCount("50%")
        result = count(SAMPLE)
        assert len(result) == 5

    def test_100_percent(self):
        count = ItemsCount("100%")
        result = count(SAMPLE)
        assert len(result) == 10

    def test_zero_percent_returns_at_least_one(self):
        """0% should still return at least 1 sentence."""
        count = ItemsCount("0%")
        result = count(SAMPLE)
        assert len(result) >= 1

    def test_10_percent_on_small_list(self):
        """10% of 3 items = 0.3, rounded to at least 1."""
        count = ItemsCount("10%")
        result = count([1, 2, 3])
        assert len(result) >= 1

    def test_repr_percentage(self):
        count = ItemsCount("20%")
        assert "20%" in repr(count)


class TestItemsCountString:
    """ItemsCount with numeric string values."""

    def test_string_count_5(self):
        count = ItemsCount("5")
        result = count(SAMPLE)
        assert len(result) == 5
        assert result == [0, 1, 2, 3, 4]

    def test_string_count_1(self):
        count = ItemsCount("1")
        result = count(SAMPLE)
        assert len(result) == 1
        assert result == [0]

    def test_string_count_3(self):
        count = ItemsCount("3")
        result = count(SAMPLE)
        assert len(result) == 3

    def test_string_count_exceeds_length(self):
        """Count larger than sequence returns all items."""
        count = ItemsCount("100")
        result = count(SAMPLE)
        assert len(result) == len(SAMPLE)


class TestItemsCountInteger:
    """ItemsCount with integer values."""

    def test_integer_3(self):
        count = ItemsCount(3)
        result = count(SAMPLE)
        assert len(result) == 3
        assert result == [0, 1, 2]

    def test_integer_1(self):
        count = ItemsCount(1)
        result = count(SAMPLE)
        assert len(result) == 1

    def test_integer_0(self):
        count = ItemsCount(0)
        result = count(SAMPLE)
        assert len(result) == 0

    def test_integer_exceeds_length(self):
        count = ItemsCount(100)
        result = count(SAMPLE)
        assert len(result) == len(SAMPLE)

    def test_float_value(self):
        """Float value is truncated to int."""
        count = ItemsCount(3.7)
        result = count(SAMPLE)
        assert len(result) == 3


class TestItemsCountBoundary:
    """ItemsCount boundary and edge cases."""

    def test_empty_sequence_percentage(self):
        count = ItemsCount("50%")
        result = count([])
        assert result == []

    def test_empty_sequence_count(self):
        count = ItemsCount(3)
        result = count([])
        assert result == []

    def test_single_item_sequence(self):
        count = ItemsCount("50%")
        result = count(["only"])
        assert len(result) >= 1

    def test_repr_integer(self):
        count = ItemsCount(5)
        assert "5" in repr(count)

    def test_repr_string(self):
        count = ItemsCount("3")
        assert "3" in repr(count)
