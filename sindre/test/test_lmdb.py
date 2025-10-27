"""
sindre.lmdb模块测试用例
测试LMDB数据库的读写操作和各种功能
"""

import pytest
import os
import numpy as np
from sindre.lmdb.pylmdb import Reader, Writer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LMDB_FILE = os.path.join(DATA_DIR, "test.db")
WRITE_FILE = os.path.join(DATA_DIR, "test_write.db")

@pytest.mark.skipif(not os.path.exists(LMDB_FILE), reason="test.db不存在")
class TestLMDBReader:
    def test_reader_basic(self):
        reader = Reader(LMDB_FILE, multiprocessing=False)
        assert len(reader) > 0
        sample = reader[0]
        assert isinstance(sample, dict)
        keys = reader.get_data_keys(0)
        assert isinstance(keys, list)
        reader.close()

    def test_reader_context(self):
        with Reader(LMDB_FILE) as reader:
            assert len(reader) > 0
            assert isinstance(reader[0], dict)

    def test_reader_get_samples(self):
        reader = Reader(LMDB_FILE)
        samples = reader.get_samples(0, size=2)
        assert isinstance(samples, list)
        reader.close()

    def test_reader_get_data_value(self):
        reader = Reader(LMDB_FILE)
        keys = reader.get_data_keys(0)
        for k in keys:
            v = reader.get_data_value(0, k)
            assert v is not None
        reader.close()

@pytest.mark.skipif(os.path.exists(WRITE_FILE), reason="test_write.db已存在，请先手动删除")
class TestLMDBWriter:
    def test_writer_basic(self):
        writer = Writer(WRITE_FILE, map_size_limit=1024*1024*10)
        data = {0: {"arr": np.arange(10), "label": 1}}
        writer.put_sample(data)
        writer.close()
        assert os.path.exists(WRITE_FILE)
        # 读回校验
        reader = Reader(WRITE_FILE)
        assert len(reader) == 1
        arr = reader.get_data_value(0, "arr")
        assert np.all(arr == np.arange(10))
        reader.close()
        os.remove(WRITE_FILE)


if __name__ == "__main__":
    pytest.main([__file__]) 