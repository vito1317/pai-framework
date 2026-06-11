"""`.pai` — Proactive AI 打包容器格式（類 GGUF 的二進位單檔格式）。

把一個完整的主動式 AI agent（manifest、policy、rules、triggers、actions、
記憶快照、甚至嵌入的模型權重如 .gguf）打包成單一可發布檔案。

=== .pai Binary Format v1（全部 little-endian）===

[Header — 32 bytes]
  offset  size  field
  0       4     magic        = b"PAI\\x01"
  4       4     version      = uint32 (目前 1)
  8       4     n_sections   = uint32
  12      4     flags        = uint32 (bit0: 1=所有段 zlib 壓縮)
  16      8     created_at   = uint64 (unix epoch 秒)
  24      8     reserved

[Section Table — n_sections 筆，緊接 header]
  每筆：
    name_len     uint16
    name         UTF-8 bytes（如 "manifest.json", "weights.gguf"）
    stype        uint32 (0=json, 1=binary, 2=sqlite, 3=text, 4=gguf)
    compressed   uint8  (0/1)
    offset       uint64 （檔案絕對位移，64-byte 對齊）
    size         uint64 （壓縮後實際儲存大小）
    raw_size     uint64 （解壓後大小）
    sha256       32 bytes（解壓後內容雜湊）

[Data Area]
  各段資料，每段起點 64-byte 對齊（padding 以 \\x00 填充）。

慣例段名：
  manifest.json   必要。agent 名稱/版本/作者/說明/相容的 pai 框架版本
  policy.json     ProactivityPolicy 參數
  triggers.json   觸發器宣告（type + 參數）
  rules.json      RuleBrain 宣告式規則（條件/動作/等級）
  actions.json    動作宣告（type + 參數，如 webhook url）
  brain.json      決策腦設定（rule / llm、模型名、system prompt）
  memory.db       選用。SQLite 記憶快照
  weights.gguf    選用。嵌入的本地模型權重
"""
from __future__ import annotations

import hashlib
import io
import json
import mmap
import os
import struct
import tempfile
import time
import zlib
from dataclasses import dataclass
from typing import Optional

_CHUNK = 8 * 1024 * 1024  # 串流讀寫塊大小 8 MiB

MAGIC = b"PAI\x01"
FORMAT_VERSION = 1
ALIGN = 64

STYPE_JSON, STYPE_BINARY, STYPE_SQLITE, STYPE_TEXT, STYPE_GGUF = 0, 1, 2, 3, 4

_STYPE_BY_NAME = {
    ".json": STYPE_JSON, ".db": STYPE_SQLITE, ".sqlite": STYPE_SQLITE,
    ".txt": STYPE_TEXT, ".md": STYPE_TEXT, ".py": STYPE_TEXT, ".gguf": STYPE_GGUF,
}


def _guess_stype(name: str) -> int:
    for ext, st in _STYPE_BY_NAME.items():
        if name.endswith(ext):
            return st
    return STYPE_BINARY


@dataclass
class Section:
    name: str
    stype: int
    data: Optional[bytes] = None       # 小段：直接持有內容
    src_path: Optional[str] = None     # 大段：路徑引用，寫檔時串流複製（不進 RAM）
    compressed: bool = True


def _hash_file(path: str) -> tuple[bytes, int]:
    """串流計算檔案 SHA-256 與大小。"""
    h, size = hashlib.sha256(), 0
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
            size += len(chunk)
    return h.digest(), size


class PaiWriter:
    """組裝並寫出 .pai 檔案（大段以串流寫入，記憶體占用恆定）。"""

    def __init__(self, compress: bool = True):
        self.compress = compress
        self.sections: list[Section] = []

    def add_json(self, name: str, obj) -> "PaiWriter":
        data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.sections.append(Section(name, STYPE_JSON, data=data,
                                     compressed=self.compress))
        return self

    def add_bytes(self, name: str, data: bytes, stype: Optional[int] = None,
                  compress: Optional[bool] = None) -> "PaiWriter":
        self.sections.append(Section(
            name, stype if stype is not None else _guess_stype(name), data=data,
            compressed=self.compress if compress is None else compress))
        return self

    def add_file(self, name: str, path: str, compress: Optional[bool] = None,
                 stream_threshold: int = 32 * 1024 * 1024) -> "PaiWriter":
        """加入檔案。超過 stream_threshold（預設 32MB）自動改為串流引用且不壓縮。"""
        size = os.path.getsize(path)
        if size >= stream_threshold:
            self.sections.append(Section(name, _guess_stype(name),
                                         src_path=path, compressed=False))
            return self
        with open(path, "rb") as f:
            return self.add_bytes(name, f.read(), compress=compress)

    def write(self, path: str) -> str:
        if not any(s.name == "manifest.json" for s in self.sections):
            raise ValueError(".pai 檔案必須包含 manifest.json 段")

        # 準備每段的 (stored_size, raw_size, sha256)；大段串流計算
        metas = []
        payload_cache: dict[int, bytes] = {}
        for i, s in enumerate(self.sections):
            if s.src_path is not None:
                sha, size = _hash_file(s.src_path)
                metas.append((size, size, sha))
            else:
                blob = zlib.compress(s.data, 9) if s.compressed else s.data
                payload_cache[i] = blob
                metas.append((len(blob), len(s.data), hashlib.sha256(s.data).digest()))

        table_size = sum(2 + len(s.name.encode()) + 4 + 1 + 8 + 8 + 8 + 32
                         for s in self.sections)
        data_start = 32 + table_size

        offsets, cursor = [], data_start
        for stored, _raw, _sha in metas:
            cursor = (cursor + ALIGN - 1) // ALIGN * ALIGN
            offsets.append(cursor)
            cursor += stored

        with open(path, "wb") as out:
            out.write(MAGIC)
            out.write(struct.pack("<III", FORMAT_VERSION, len(self.sections),
                                  1 if self.compress else 0))
            out.write(struct.pack("<Q", int(time.time())))
            out.write(b"\x00" * 8)

            for s, (stored, raw, sha), off in zip(self.sections, metas, offsets):
                nb = s.name.encode()
                out.write(struct.pack("<H", len(nb)))
                out.write(nb)
                out.write(struct.pack("<IB", s.stype, 1 if s.compressed else 0))
                out.write(struct.pack("<QQQ", off, stored, raw))
                out.write(sha)

            for i, (s, off) in enumerate(zip(self.sections, offsets)):
                out.write(b"\x00" * (off - out.tell()))
                if s.src_path is not None:
                    with open(s.src_path, "rb") as f:
                        while chunk := f.read(_CHUNK):
                            out.write(chunk)
                else:
                    out.write(payload_cache[i])
        return path


class PaiReader:
    """讀取與驗證 .pai 檔案（mmap，開檔零拷貝，大段以串流抽取）。"""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        self._raw = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self._parse()

    def _parse(self):
        raw = self._raw
        if raw[:4] != MAGIC:
            raise ValueError("不是有效的 .pai 檔案（magic 不符）")
        self.version, n_sections, self.flags = struct.unpack_from("<III", raw, 4)
        if self.version > FORMAT_VERSION:
            raise ValueError(f"不支援的 .pai 版本：{self.version}")
        (self.created_at,) = struct.unpack_from("<Q", raw, 16)

        self._index: dict[str, dict] = {}
        pos = 32
        for _ in range(n_sections):
            (name_len,) = struct.unpack_from("<H", raw, pos); pos += 2
            name = raw[pos:pos + name_len].decode(); pos += name_len
            stype, compressed = struct.unpack_from("<IB", raw, pos); pos += 5
            offset, size, raw_size = struct.unpack_from("<QQQ", raw, pos); pos += 24
            sha = raw[pos:pos + 32]; pos += 32
            self._index[name] = dict(stype=stype, compressed=compressed,
                                     offset=offset, size=size,
                                     raw_size=raw_size, sha256=sha)

    @property
    def section_names(self) -> list[str]:
        return list(self._index)

    def read_bytes(self, name: str, verify: bool = True) -> bytes:
        meta = self._index[name]
        blob = self._raw[meta["offset"]:meta["offset"] + meta["size"]]
        data = zlib.decompress(blob) if meta["compressed"] else blob
        if len(data) != meta["raw_size"]:
            raise ValueError(f"段 '{name}' 大小不符")
        if verify and hashlib.sha256(data).digest() != meta["sha256"]:
            raise ValueError(f"段 '{name}' SHA-256 校驗失敗（檔案損毀或被竄改）")
        return data

    def read_json(self, name: str):
        return json.loads(self.read_bytes(name))

    def extract_to(self, name: str, dest: str, verify: bool = True) -> str:
        """把大段串流寫到 dest（分塊，記憶體占用恆定），邊寫邊校驗 SHA-256。"""
        meta = self._index[name]
        if meta["compressed"]:
            data = self.read_bytes(name, verify=verify)   # 壓縮段通常很小
            with open(dest, "wb") as f:
                f.write(data)
            return dest
        h = hashlib.sha256()
        with open(dest, "wb") as f:
            pos, end = meta["offset"], meta["offset"] + meta["size"]
            while pos < end:
                chunk = self._raw[pos:min(pos + _CHUNK, end)]
                h.update(chunk)
                f.write(chunk)
                pos += len(chunk)
        if verify and h.digest() != meta["sha256"]:
            os.remove(dest)
            raise ValueError(f"段 '{name}' SHA-256 校驗失敗（檔案損毀或被竄改）")
        return dest

    def extract_to_cache(self, name: str, verify: bool = True) -> str:
        """抽取到共用快取（以段的 SHA-256 命名）；已存在則直接重用，零成本。"""
        meta = self._index[name]
        cache_dir = os.path.join(tempfile.gettempdir(), "pai_weights_cache")
        os.makedirs(cache_dir, exist_ok=True)
        ext = os.path.splitext(name)[1] or ".bin"
        dest = os.path.join(cache_dir, meta["sha256"].hex()[:16] + ext)
        if os.path.exists(dest) and os.path.getsize(dest) == meta["raw_size"]:
            return dest
        return self.extract_to(name, dest, verify=verify)

    @property
    def manifest(self) -> dict:
        return self.read_json("manifest.json")

    def info(self) -> dict:
        return {
            "path": self.path,
            "format_version": self.version,
            "created_at": self.created_at,
            "sections": {
                n: {"type": m["stype"], "stored_bytes": m["size"],
                    "raw_bytes": m["raw_size"]}
                for n, m in self._index.items()
            },
        }


# ---- 高階 API：打包/載入整個 agent ----

def pack_agent(path: str, manifest: dict, policy: dict, triggers: list,
               rules: list, actions: dict, brain: Optional[dict] = None,
               memory_db: Optional[str] = None,
               weights_path: Optional[str] = None) -> str:
    """把 agent 定義打包成單一 .pai 檔案。"""
    w = PaiWriter()
    w.add_json("manifest.json", {
        "pai_format": FORMAT_VERSION, "kind": "proactive-agent", **manifest})
    w.add_json("policy.json", policy)
    w.add_json("triggers.json", triggers)
    w.add_json("rules.json", rules)
    w.add_json("actions.json", actions)
    if brain:
        w.add_json("brain.json", brain)
    if memory_db:
        w.add_file("memory.db", memory_db)
    if weights_path:
        # 模型權重通常已壓縮/量化，不再 zlib（避免白費時間）
        w.add_file("weights.gguf", weights_path, compress=False)
    return w.write(path)


def load_agent(path: str) -> dict:
    """載入 .pai 檔案，回傳完整 agent 定義 dict。"""
    r = PaiReader(path)
    out = {"manifest": r.manifest, "info": r.info()}
    for name in ("policy.json", "triggers.json", "rules.json",
                 "actions.json", "brain.json"):
        if name in r.section_names:
            out[name.replace(".json", "")] = r.read_json(name)
    out["has_memory"] = "memory.db" in r.section_names
    out["has_weights"] = "weights.gguf" in r.section_names
    return out
