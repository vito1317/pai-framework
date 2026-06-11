"""共用工具：穩健的 JSON 陣列抽取 + LLM HTTP 重試退避。

取代各決策腦原本的 find("[")..rfind("]") 貪婪解析——後者在模型輸出
多個 JSON 陣列、或陣列前後夾帶說明文字/markdown code fence 時會解析錯誤。
這裡用「括號配對掃描」找出第一個完整、平衡的頂層 JSON 陣列，且正確跳過
字串內的括號與跳脫字元。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

logger = logging.getLogger("pai.jsonutil")


def extract_json_array(text: str) -> list:
    """從任意文字中抽出第一個完整、平衡的頂層 JSON 陣列並解析。

    - 正確處理字串內的 [ ] { } 與跳脫（如 "a\\"b[" ）
    - 容許陣列前後有說明文字 / ```json fence
    - 找不到合法陣列時回傳 []
    """
    start = text.find("[")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, list):
                            return parsed
                    except json.JSONDecodeError:
                        break  # 這個起點解不出來，換下一個 [
        start = text.find("[", start + 1)
    return []


def retry_call(fn, attempts: int = 2, base_delay: float = 0.8,
               exceptions: tuple = (Exception,)):
    """對暫時性失敗（多為網路）做重試＋指數退避。

    attempts=2 表示最多嘗試 2 次（1 次重試）。全部失敗時拋出最後一個例外，
    由呼叫端決定是否退回 fallback。
    """
    last = None
    for i in range(attempts):
        try:
            return fn()
        except exceptions as exc:  # noqa: BLE001
            last = exc
            if i < attempts - 1:
                delay = base_delay * (2 ** i)
                logger.warning("call failed (%s), retry %d/%d in %.1fs",
                               exc, i + 1, attempts - 1, delay)
                time.sleep(delay)
    raise last
