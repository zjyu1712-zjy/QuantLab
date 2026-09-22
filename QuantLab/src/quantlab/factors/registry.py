"""显式管理因子类型；每个注册表独立，不自动发现或运行因子。"""

from inspect import isabstract

from quantlab.factors.base import BaseFactor, FactorDefinition


class FactorRegistry:
    """保存因子名称与因子类型之间的映射。"""

    def __init__(self) -> None:
        self._factor_types: dict[str, type[BaseFactor]] = {}

    def register(self, factor_type: type[BaseFactor]) -> None:
        """注册一个因子类型，但不计算或运行因子。"""
        if not isinstance(factor_type, type) or not issubclass(factor_type, BaseFactor):
            raise ValueError("注册项必须是因子类型")
        if isabstract(factor_type):
            raise ValueError("不能注册抽象因子类型")

        name = getattr(factor_type, "name", None)
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("因子名称必须为非空且无两端空白的字符串")
        definition = getattr(factor_type, "definition", None)
        if not isinstance(definition, FactorDefinition) or definition.name != name:
            raise ValueError("因子定义或名称不一致")

        if name in self._factor_types:
            raise ValueError(f"因子已注册：{name}")

        self._factor_types[name] = factor_type

    def get(self, name: str) -> type[BaseFactor]:
        """按名称取得已注册的因子类型。"""
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("因子名称必须为非空且无两端空白的字符串")
        try:
            return self._factor_types[name]
        except KeyError as error:
            raise KeyError(f"因子未注册：{name}") from error

    def names(self) -> tuple[str, ...]:
        """返回全部已注册因子名称。"""
        return tuple(sorted(self._factor_types))
