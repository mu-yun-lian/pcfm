"""atomic_write_json 唯一临时文件名 + 失败清理的回归测试。"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from pcfm.persistence.atomic import atomic_write_json


class AtomicWriteTests(unittest.TestCase):
    def test_roundtrip_and_no_tmp_leftover(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "x.json"
            atomic_write_json(path, {"a": 1, "b": [1, 2]})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1, "b": [1, 2]})
            self.assertEqual([p.name for p in path.parent.glob("*.tmp")], [])

    def test_cleans_tmp_on_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "x.json"
            with mock.patch("pcfm.persistence.atomic.os.replace", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    atomic_write_json(path, {"a": 1})
            # 失败后临时文件已清理
            self.assertEqual([p.name for p in path.parent.glob("*.tmp")], [])

    def test_concurrent_writes_to_different_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            errors: list[BaseException] = []

            def writer(i: int) -> None:
                try:
                    for _ in range(20):
                        atomic_write_json(root / f"f{i}.json", {"n": i})
                except BaseException as error:
                    errors.append(error)

            threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            for i in range(20):
                self.assertEqual(
                    json.loads((root / f"f{i}.json").read_text(encoding="utf-8")), {"n": i}
                )


if __name__ == "__main__":
    unittest.main()
