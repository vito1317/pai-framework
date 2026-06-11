# 發佈到 PyPI

PyPI 發佈名為 **`paigent`**（`pai-framework` 與既有專案太相似被 PyPI 擋下）。
import 套件名仍是 `import pai`，CLI 仍是 `pai`——只有 `pip install` 的名稱不同。

套件已建置並通過 `twine check`，產物在 `dist/`：
- `paigent-0.1.0-py3-none-any.whl`
- `paigent-0.1.0.tar.gz`

## 安裝

```bash
pip install paigent                 # 核心（零依賴）
pip install "paigent[local]"        # + llama-cpp-python（LocalLLMBrain）
pip install "paigent[omni]"         # + transformers/torch（MiniCPMoBrain）
pip install "paigent[finetune]"     # + LLaMA-Factory
pai                                  # CLI 進入點
```

## 重新建置

```bash
python3 -m build
python3 -m twine check dist/*
```

## 發佈（需要你的 PyPI API token）

在 https://pypi.org/manage/account/token/ 建立 token 後，由你本人執行
（token 完全不經過協作流程）：

```bash
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-xxxxxxxx \
  python3 -m twine upload --verbose dist/*
```

發佈後：`pip install paigent`
