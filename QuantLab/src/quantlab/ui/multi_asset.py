"""研究页面的请求身份与提示；计算和磁盘读写仍由数据层负责。"""

from dataclasses import asdict
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re

from quantlab.data.calendar import CalendarInput
from quantlab.data.research_sets import DatasetSelection


def _json_value(value):
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("研究配置包含不支持的字段类型")


def research_request_sha256(
    *, name: str, selections: tuple[DatasetSelection, ...], start: date, end: date,
    calendar: CalendarInput | None,
) -> str:
    """同样的配置有同样指纹；不读取源文件，不把当前预览替换为新版本。"""
    payload = {
        "name": name, "start": start, "end": end,
        "selections": [asdict(selection) for selection in sorted(
            selections, key=lambda item: (item.record.symbol, item.record.version_id),
        )],
        "calendar": asdict(calendar) if calendar is not None else None,
    }
    serialized = json.dumps(payload, default=_json_value, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":"), allow_nan=False)
    return sha256(serialized.encode("utf-8")).hexdigest()


def can_save_preview(current_request_sha256: str, preview_request_sha256: str | None) -> bool:
    """按钮状态和保存事件共用的检查，空值不能因彼此相等而视为有效。"""
    return (isinstance(current_request_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", current_request_sha256) is not None
            and current_request_sha256 == preview_request_sha256)


def friendly_research_error(error: Exception) -> tuple[str, str]:
    """主提示不暴露路径；详细原因仅由页面放入高级说明，不显示堆栈。"""
    if isinstance(error, FileNotFoundError):
        message = "找不到所选数据版本，请检查文件是否移动，并重新选择或导入。"
    elif isinstance(error, ValueError):
        message = "数据或声明未通过校验，请检查版本、复权口径、日期与日历，或查看详细原因。"
    elif isinstance(error, OSError):
        message = "本地文件读写失败，请检查目录权限和磁盘空间；原预览仍保留，可稍后重试。"
    else:
        message = "研究集合操作未完成，请检查输入后重试。"
    return message, str(error)
