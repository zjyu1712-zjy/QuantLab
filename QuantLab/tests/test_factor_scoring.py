"""横截面评分使用手算期望；不读取行情、不借助被测算法生成答案。"""

import numpy as np
import pandas as pd
import pytest

from quantlab import factors


def score(values, *, ascending=False):
    assert hasattr(factors, "cross_sectional_rank_scores"), "尚未实现横截面排名评分"
    return factors.cross_sectional_rank_scores(values, ascending=ascending)


@pytest.mark.parametrize("values, ascending, expected", [
    ([30, 20, 10], False, [100, 50, 0]),
    ([30, 20, 10], True, [0, 50, 100]),
    ([20, 20, 10], False, [75, 75, 0]),
    ([20, 20, 10], True, [25, 25, 100]),
    ([30, 20, 20, 10], False, [100, 50, 50, 0]),
    ([0, 0, 0], False, [50, 50, 50]),
    ([-3, -3], True, [50, 50]),
    ([np.nan, -2, 0, 2], False, [np.nan, 0, 50, 100]),
    ([1, np.nan], False, [np.nan, np.nan]),
    ([np.nan, np.nan], True, [np.nan, np.nan]),
    ([], False, []),
    ([.100001, .100002], False, [0, 100]),
    ([0, -0.0, -1], False, [75, 75, 0]),
    ([np.finfo(float).max, 1, np.nextafter(0.0, 1.0)], False, [100, 50, 0]),
])
def test_scores_follow_direction_and_average_ties_without_using_price_distances(values, ascending, expected):
    actual = score(pd.Series(values, dtype="float64"), ascending=ascending)
    pd.testing.assert_series_equal(actual, pd.Series(expected, dtype="float64", name="relative_score"))


def test_scores_preserve_duplicate_multiindex_and_do_not_modify_or_alias_input():
    index = pd.MultiIndex.from_tuples([("a", 2), ("a", 2), ("b", 1), ("c", 0)])
    values = pd.Series([2., 2., 0., np.nan], index=index, name="factor_value")
    original = values.copy(deep=True)
    actual = score(values)
    pd.testing.assert_series_equal(actual, pd.Series([75., 75., 0., np.nan], index=index, name="relative_score"))
    actual.iloc[0] = -999
    pd.testing.assert_series_equal(values, original)


def test_average_ties_reverse_symmetrically_and_score_is_order_invariant():
    values = pd.Series([8., 3., 3., -1., np.nan], index=["d", "c", "a", "b", "e"])
    high = score(values)
    low = score(values, ascending=True)
    np.testing.assert_allclose((high + low).dropna(), [100., 100., 100., 100.])
    permuted = score(values.iloc[[4, 2, 0, 3, 1]])
    pd.testing.assert_series_equal(permuted.reindex(values.index), high)


def test_internal_scores_keep_precision_and_only_depend_on_relative_order():
    values = pd.Series([4., 3., 2., 1.])
    actual = score(values)
    assert actual.iloc[1] == pytest.approx(200 / 3)
    assert actual.iloc[1] != round(actual.iloc[1], 2)
    stretched = score(pd.Series([40000., 300., 2., .001]))
    pd.testing.assert_series_equal(actual, stretched)


@pytest.mark.parametrize("values", [
    [1., 2.], pd.DataFrame({"value": [1., 2.]}),
    pd.Series(["1", "2"]), pd.Series([True, False]),
    pd.Series([1 + 1j, 2 + 0j]),
    pd.Series([1., np.inf]), pd.Series([-np.inf, np.nan]),
])
def test_rejects_non_float_series_or_infinity_instead_of_silently_scoring(values):
    with pytest.raises(ValueError):
        score(values)


@pytest.mark.parametrize("ascending", [None, 0, "ascending"])
def test_requires_an_explicit_boolean_direction(ascending):
    with pytest.raises(ValueError):
        score(pd.Series([1., 2.]), ascending=ascending)
