"""比較 Python 程式結構，讓純註解／排版修改不會阻擋舊結果的沿用。"""

import ast
from pathlib import Path


def same_python_structure(saved_path: Path, current_path: Path) -> bool:
    """只忽略註解與排版，保留常數、運算、呼叫順序及文件字串的差異。

    這不是任意兩段程式的行為等價判定。變數改名、改參數值或重排邏輯
    都仍會視為不同；原始結果的 SHA-256 完整性檢查也必須另外保留。
    """
    saved_tree = ast.parse(Path(saved_path).read_text(encoding="utf-8"))
    current_tree = ast.parse(Path(current_path).read_text(encoding="utf-8"))
    return ast.dump(saved_tree, include_attributes=False) == ast.dump(
        current_tree, include_attributes=False
    )
