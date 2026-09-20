#!/bin/bash

# =========================
# 讀取 config.json
# =========================

ALPHA=$(python -c "import json; print(json.load(open('config.json'))['alpha'])")
ENABLE_POISON=$(python -c "import json; print(json.load(open('config.json'))['enable_poison'])")
POISON_SCALE=$(python -c "import json; print(json.load(open('config.json'))['poison_scale'])")



# =========================
# 決定 CSV 檔名
# =========================

if [ "$ENABLE_POISON" = "True" ]; then

    FILENAME="alpha_${ALPHA}_poisoning_scale_${POISON_SCALE}.csv"

else

    FILENAME="alpha_${ALPHA}.csv"

fi


echo "=============================="
echo "Experiment Settings"
echo "ALPHA          = $ALPHA"
echo "Poisoning      = $ENABLE_POISON"
echo "Poison Scale   = $POISON_SCALE"
echo "CSV            = $FILENAME"
echo "=============================="


# =========================
# 執行 run.sh
# =========================

./run.sh


# =========================
# 修改 results.csv 檔名
# =========================

mv results.csv "$FILENAME"

echo ""
echo "Result saved as:"
echo "$FILENAME"
