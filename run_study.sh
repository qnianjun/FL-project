#!/usr/bin/env bash
# 實驗 A/B 的 shell 入口：預設只預覽，加 --run 才開始訓練。
# 發生錯誤立即停止，避免在前一步失敗後繼續執行。
set -euo pipefail
# 切回腳本所在的專案目錄，確保 config.json 與 data/ 路徑正確。
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# 固定使用專案虛擬環境，並原樣傳遞你輸入的命令列參數。
exec .venv/bin/python run_study.py "$@"
