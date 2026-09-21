"""確保可讀性修改不阻擋沿用，但真正的程式變更仍會被拒絕。"""
import tempfile
from pathlib import Path
import unittest

from source_compatibility import same_python_structure


class SourceCompatibilityTests(unittest.TestCase):
    def compare(self, first, second):
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / 'saved.py'
            current = Path(directory) / 'current.py'
            saved.write_text(first)
            current.write_text(second)
            return same_python_structure(saved, current)

    def test_comments_and_formatting_are_allowed(self):
        self.assertTrue(self.compare('scale=10\ny=x*scale\n', '# 放大更新\nscale = 10\ny = (x * scale)  # 保留運算\n'))

    def test_parameter_operator_and_order_changes_are_rejected(self):
        original = 'scale=10\ny=x*scale\ntrain()\nevaluate()\n'
        for changed in [original.replace('10', '20'), original.replace('*', '+'),
                        original.replace('train()\nevaluate()', 'evaluate()\ntrain()')]:
            with self.subTest(changed=changed):
                self.assertFalse(self.compare(original, changed))

    def test_strings_and_docstrings_are_not_ignored(self):
        self.assertFalse(self.compare('"""first"""\nx="clean"\n', '"""first"""\nx="poisoned"\n'))
        self.assertFalse(self.compare('"""first"""\nx=1\n', '"""second"""\nx=1\n'))


if __name__ == '__main__':
    unittest.main()
