"""第三部分圖表的繁體中文字型設定。"""
from __future__ import annotations

from matplotlib import font_manager, rcParams


def configure_chinese_font() -> str:
    """選擇可用中文字型，避免繁體中文圖表變成缺字方框。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ['PingFang TC','Noto Sans CJK TC','Noto Sans TC','Microsoft JhengHei','Heiti TC','Arial Unicode MS']:
        if name in available:
            rcParams['font.family'] = [name,'DejaVu Sans']
            rcParams['axes.unicode_minus'] = False
            return name
    raise RuntimeError('找不到中文字型。請安裝思源黑體、Noto Sans CJK TC 或其他支援繁體中文的字型，再重新產生圖表。')
