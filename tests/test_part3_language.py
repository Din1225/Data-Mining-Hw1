"""確認來源紀錄可在 repository 搬移後使用。"""
import sys
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from analysis_part3 import provenance


class LanguageTests(unittest.TestCase):
    def test_provenance_can_be_resolved_after_repository_is_moved(self):
        root = Path(__file__).resolve().parents[1]
        source = root / 'src/part3_chinese.py'
        with tempfile.TemporaryDirectory() as directory:
            provenance(Path(directory), [source], {})
            record = json.loads((Path(directory) / 'provenance.json').read_text())
            self.assertEqual(record['path_base'], 'repository_root')
            self.assertEqual(record['inputs'][0]['path'], 'src/part3_chinese.py')
            resolved = root / record['inputs'][0]['path']
            self.assertEqual(hashlib.sha256(resolved.read_bytes()).hexdigest(),
                             record['inputs'][0]['sha256'])



if __name__ == '__main__':
    unittest.main()
