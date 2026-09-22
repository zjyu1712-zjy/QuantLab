"""因子接口与显式注册的契约；仅用内存合成数据，不执行投资公式。"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quantlab.factors import BaseFactor, FactorDefinition, FactorEvaluation, FactorRegistry


DEFINITION = FactorDefinition(
    name="test_factor",
    description="接口测试",
    formula="constant 0",
    version="1.0",
    required_fields=("close", "has_observation"),
    lookback_periods=20,
    window_observations=21,
    direction="higher_is_stronger",
    missing_policy="strict_window",
    availability="after_close",
)


class TestOnlyFactor(BaseFactor):
    __test__ = False
    name = "test_factor"
    definition = DEFINITION

    def evaluate(self, market_data):
        return FactorEvaluation(
            pd.Series(0.0, index=market_data.index, dtype="float64"),
            pd.Series("valid", index=market_data.index, dtype="string"),
        )


def test_compute_keeps_series_interface_and_index():
    frame = pd.DataFrame({"close": [10.0, 11.0]}, index=pd.Index([3, 9], name="row"))
    factor = TestOnlyFactor()
    actual = factor.compute(frame)
    expected = pd.Series([0.0, 0.0], index=frame.index, dtype="float64")
    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_series_equal(actual, factor.evaluate(frame).values)
    assert actual.index.identical(frame.index)


def test_compute_delegates_once_and_returns_an_independent_values_copy():
    class StoredResultFactor(TestOnlyFactor):
        def __init__(self):
            self.calls = 0
            self.output = FactorEvaluation(
                pd.Series([np.nan, 0.1], index=[3, 9], dtype="float64"),
                pd.Series(["warmup", "valid"], index=[3, 9], dtype="string"),
            )

        def evaluate(self, market_data):
            self.calls += 1
            return self.output

    factor = StoredResultFactor()
    frame = pd.DataFrame({"close": [10.0, 11.0]}, index=[3, 9])
    actual = factor.compute(frame)
    assert factor.calls == 1
    pd.testing.assert_series_equal(
        actual, pd.Series([np.nan, 0.1], index=[3, 9], dtype="float64"),
    )
    actual.iloc[1] = 99.0
    assert factor.output.values.iloc[1] == 0.1
    assert factor.output.status.tolist() == ["warmup", "valid"]


def test_registry_is_explicit_independent_and_duplicate_safe():
    left, right = FactorRegistry(), FactorRegistry()
    assert left.names() == right.names() == ()
    left.register(TestOnlyFactor)
    assert left.get("test_factor") is TestOnlyFactor
    assert left.names() == ("test_factor",) and right.names() == ()
    with pytest.raises(ValueError, match="重复|已注册"):
        left.register(TestOnlyFactor)
    with pytest.raises(KeyError, match="未注册"):
        right.get("test_factor")
    assert left.get("test_factor") is TestOnlyFactor


def test_registry_returns_sorted_names_without_instantiating_or_evaluating():
    class NeverRunFactor(TestOnlyFactor):
        def __init__(self):
            raise AssertionError("注册或查询不能实例化因子")

        def evaluate(self, market_data):
            raise AssertionError("注册或查询不能执行因子")

    registry = FactorRegistry()
    for name in ("zeta", "alpha"):
        factor_type = type(name, (NeverRunFactor,), {
            "name": name, "definition": replace(DEFINITION, name=name),
        })
        registry.register(factor_type)
        assert registry.get(name) is factor_type
    assert registry.names() == ("alpha", "zeta")


@pytest.mark.parametrize("bad", [None, 123, object, BaseFactor, TestOnlyFactor()])
def test_registry_rejects_nonconcrete_factor_types(bad):
    with pytest.raises(ValueError, match="因子|类型|抽象"):
        FactorRegistry().register(bad)


def test_registry_rejects_named_abstract_factor():
    class AbstractFactor(BaseFactor):
        name = "test_factor"
        definition = DEFINITION

    with pytest.raises(ValueError, match="抽象"):
        FactorRegistry().register(AbstractFactor)


@pytest.mark.parametrize("name", [None, 1, "", " ", " test_factor "])
def test_registry_rejects_invalid_class_names(name):
    bad = type("Bad", (TestOnlyFactor,), {"name": name})
    registry = FactorRegistry()
    with pytest.raises(ValueError, match="名称"):
        registry.register(bad)
    assert registry.names() == ()


@pytest.mark.parametrize("definition", [None, {}, replace(DEFINITION, name="different")])
def test_registry_rejects_mismatched_or_missing_definition(definition):
    bad = type("Bad", (TestOnlyFactor,), {"definition": definition})
    registry = FactorRegistry()
    with pytest.raises(ValueError, match="定义|名称"):
        registry.register(bad)
    assert registry.names() == ()


@pytest.mark.parametrize("name", [None, [], 1, "", " ", " test_factor "])
def test_registry_invalid_lookup_name_has_clear_error(name):
    registry = FactorRegistry()
    registry.register(TestOnlyFactor)
    with pytest.raises(ValueError, match="名称"):
        registry.get(name)


@pytest.mark.parametrize("field", [
    "name", "description", "formula", "version", "direction", "missing_policy", "availability",
])
@pytest.mark.parametrize("value", [None, 1, "", " \n ", " padded "])
def test_definition_rejects_invalid_metadata(field, value):
    with pytest.raises(ValueError):
        replace(DEFINITION, **{field: value})


@pytest.mark.parametrize("value", [
    [], (), "close", ("close", "close"), ("",), (" close",), (None,), ([],),
])
def test_definition_rejects_invalid_required_fields(value):
    with pytest.raises(ValueError, match="字段"):
        replace(DEFINITION, required_fields=value)


@pytest.mark.parametrize("field", ["lookback_periods", "window_observations"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 20.0, "20", np.int64(20)])
def test_definition_rejects_nonpositive_or_noninteger_periods(field, value):
    with pytest.raises(ValueError, match="周期|观测数"):
        replace(DEFINITION, **{field: value})


@pytest.mark.parametrize("observations", [1, 20, 22])
def test_definition_strict_window_includes_current_observation(observations):
    with pytest.raises(ValueError, match="窗口"):
        replace(DEFINITION, window_observations=observations)
