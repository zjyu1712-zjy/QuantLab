"""因子类型注册表的最小接口。"""

from quantlab.factors.base import BaseFactor


class FactorRegistry:
    """保存因子名称与因子类型之间的映射。"""

    def __init__(self) -> None:
        self._factor_types: dict[str, type[BaseFactor]] = {}

    def register(self, factor_type: type[BaseFactor]) -> None:
        """注册一个因子类型，但不计算或运行因子。"""
        name = getattr(factor_type, "name", "").strip()

        if not name:
            raise ValueError("因子必须定义非空名称")

        if name in self._factor_types:
            raise ValueError(f"因子已注册：{name}")

        self._factor_types[name] = factor_type

    def get(self, name: str) -> type[BaseFactor]:
        """按名称取得已注册的因子类型。"""
        try:
            return self._factor_types[name]
        except KeyError as error:
            raise KeyError(f"因子未注册：{name}") from error

    def names(self) -> tuple[str, ...]:
        """返回全部已注册因子名称。"""
        return tuple(sorted(self._factor_types))
