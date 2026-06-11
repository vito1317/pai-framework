# 發佈到 PyPI

套件已建置並通過 `twine check`，產物在 `dist/`：
- `pai_framework-0.1.0-py3-none-any.whl`
- `pai_framework-0.1.0.tar.gz`

## 安裝（本地驗證）

```bash
pip install dist/pai_framework-0.1.0-py3-none-any.whl
pai                       # CLI 進入點
# 選配額外功能
pip install "pai-framework[local]"      # llama-cpp-python
pip install "pai-framework[omni]"       # MiniCPM-o (transformers/torch)
pip install "pai-framework[finetune]"   # LLaMA-Factory
```

## 重新建置

```bash
python3 -m build           # 產生 wheel + sdist
python3 -m twine check dist/*
```

## 發佈（需要你的 PyPI API token）

`pai-framework` 這個名稱在 PyPI 上尚未被佔用。在 https://pypi.org/manage/account/token/
建立 token 後，擇一執行：

```bash
# 方式 A：環境變數（推薦，token 不進指令歷史）
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-xxxxxxxx          # 你的 token
python3 -m twine upload dist/*

# 方式 B：先上 TestPyPI 驗證流程
python3 -m twine upload --repository testpypi dist/*
```

> 基於安全，本專案的協作流程不會代為輸入金鑰；請由你本人提供 token 執行上述指令。

發佈後安裝：`pip install pai-framework`
