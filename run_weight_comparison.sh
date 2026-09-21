#!/usr/bin/env bash
# 實驗 D 的入口；預設只顯示設定，加入 --run 才開始六次訓練。
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec .venv/bin/python run_weight_comparison.py "$@"
