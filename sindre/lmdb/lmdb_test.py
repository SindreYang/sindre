import shutil
import time
import traceback

from sindre.lmdb import *




if __name__ == '__main__':
    db_path = "test.db"

    # 清理之前的测试文件
    if os.path.exists(db_path):
        if os.path.isdir(db_path):
            shutil.rmtree(db_path)
        else:
            os.remove(db_path)

    # 测试1: 创建数据库并写入数据
    print("\n1. 测试数据库写入功能")
    print("-" * 40)

    try:
        with Writer(dirpath=db_path, map_size_limit=1024) as writer:
            print(writer)

            # 创建包含嵌套字典的测试数据
            nested_dict_sample = {
                "metadata": {
                    "user_info": {
                        "name": "测试用户",
                        "age": 25,
                        "preferences": {
                            "theme": "dark",
                            "language": "zh-CN"
                        }
                    },
                    "system_info": {
                        "version": "1.0.0",
                        "settings": {
                            "quality": "high",
                            "notifications": True
                        }
                    }
                },
                "nested_arrays": {
                    "matrix": [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                    ],
                    "coordinates": {
                        "x": [1.0, 2.0, 3.0],
                        "y": [4.0, 5.0, 6.0]
                    }
                },
                "image_data": np.random.rand(28, 28, 3).astype(np.float32),
                "simple_value": 42
            }

            # 创建包含自定义对象的测试数据
            class TestObject:
                def __init__(self, name, value, data):
                    self.name = name
                    self.value = value
                    self.data = data
                    self.timestamp = time.time()

                def __repr__(self):
                    return f"TestObject(name={self.name}, value={self.value}, data_shape={getattr(self.data, 'shape', 'No shape')})"

                def get_info(self):
                    return f"Object: {self.name} with value {self.value}"

            # 创建object类型数据
            custom_object = TestObject(
                name="测试对象",
                value=100,
                data=np.random.rand(5, 5)
            )

            object_sample = {
                "custom_object": custom_object,
                "object_list": [
                    TestObject(f"obj_{i}", i, np.random.rand(3, 3))
                    for i in range(3)
                ],
                "object_dict": {
                    "first": TestObject("first", 1, np.array([1, 2, 3])),
                    "second": TestObject("second", 2, np.array([4, 5, 6]))
                },
                "mixed_data": {
                    "numpy_array": np.random.rand(10),
                    "python_list": [1, 2, 3, 4, 5],
                    "tuple_data": (1, 2, 3),
                    "set_data": {1, 2, 3}
                }
            }

            # 测试 put_sample 方法
            print("添加嵌套字典样本...")
            writer.put_sample(nested_dict_sample)

            print("添加对象样本...")
            writer.put_sample(object_sample)

            # 添加更多测试样本
            for i in range(2):
                complex_sample = {
                    "level1": {
                        "level2": {
                            "level3": {
                                "final_value": i * 100,
                                "array_data": np.random.rand(i + 1, i + 1),
                                "nested_list": [[j * k for k in range(3)] for j in range(2)]
                            }
                        },
                        "direct_array": np.arange(i * 10, (i + 1) * 10)
                    },
                    "flat_data": np.random.rand(8)
                }
                writer.put_sample(complex_sample)

            # 测试元数据存储
            writer.put_meta("dataset_info", {
                "name": "嵌套字典和对象测试数据集",
                "version": "3.0",
                "data_types": ["nested_dict", "custom_objects", "numpy_arrays"],
                "created": time.strftime("%Y-%m-%d %H:%M:%S")
            })

            writer.put_meta("object_types", {
                "custom_classes": ["TestObject"],
                "supported_types": ["dict", "list", "tuple", "set", "ndarray", "object"]
            })

        print(f"数据库写入完成，数据库大小: {writer.size}")

    except Exception as e:
        print(f"数据库写入测试失败: {e}")
        traceback.print_exc()

    # 测试2: 读取数据库数据
    print("\n2. 测试数据库读取功能")
    print("-" * 40)

    try:
        with Reader(dirpath=db_path) as reader:
            print(reader)

            # 测试嵌套字典读取
            print("\n读取嵌套字典样本:")
            sample_0 = reader[0]
            print(f"样本0的键: {list(sample_0.keys())}")

            if 'metadata' in sample_0:
                print("✓ 嵌套字典结构:")
                print(f"  - user_info.name: {sample_0['metadata']['user_info']['name']}")
                print(f"  - system_info.version: {sample_0['metadata']['system_info']['version']}")
                print(f"  - preferences.theme: {sample_0['metadata']['user_info']['preferences']['theme']}")

            # 测试对象读取
            print("\n读取对象样本:")
            sample_1 = reader[1]
            print(f"样本1的键: {list(sample_1.keys())}")

            if 'custom_object' in sample_1:
                obj = sample_1['custom_object']
                print(f"✓ 自定义对象类型: {type(obj)}")
                print(f"✓ 对象属性: name={obj.name}, value={obj.value}")
                if hasattr(obj, 'get_info'):
                    print(f"✓ 对象方法: {obj.get_info()}")

            if 'object_list' in sample_1:
                print(f"✓ 对象列表长度: {len(sample_1['object_list'])}")
                for i, item in enumerate(sample_1['object_list']):
                    print(f"  - 对象{i}: {item}")

            # 测试复杂嵌套结构
            if len(reader) > 2:
                sample_2 = reader[2]
                print(f"\n复杂嵌套样本键: {list(sample_2.keys())}")
                if 'level1' in sample_2:
                    print(f"✓ 深度嵌套值: {sample_2['level1']['level2']['level3']['final_value']}")

            # 测试元数据读取
            dataset_info = reader.get_meta("dataset_info")
            print(f"\n数据集信息: {dataset_info}")

            object_types = reader.get_meta("object_types")
            print(f"对象类型信息: {object_types}")

            # 保存第一个样本用于后续测试
            saved_sample_0 = sample_0.copy()

    except Exception as e:
        print(f"数据库读取测试失败: {e}")
        traceback.print_exc()

    # 测试3: 混合操作测试
    print("\n3. 测试混合操作")
    print("-" * 40)

    try:
        with Writer(dirpath=db_path, map_size_limit=1024) as writer:
            print(f"重新打开数据库，当前大小: {writer.size}")

            # 在现有数据基础上进行修改操作
            if writer.size > 0:
                # 使用之前保存的样本进行修改
                modified_nested_sample = saved_sample_0

                if modified_nested_sample:
                    modified_nested_sample["new_nested_field"] = {
                        "deeply": {
                            "nested": {
                                "value": "新添加的深层嵌套值",
                                "array": np.random.rand(5, 5)
                            }
                        }
                    }
                    modified_nested_sample["metadata"]["user_info"]["new_preference"] = {
                        "font_size": 14,
                        "color_scheme": "blue"
                    }

                    print("修改嵌套样本...")
                    writer.change_sample(0, modified_nested_sample, safe_model=False)

            # 插入包含混合类型的样本
            mixed_sample = {
                "types_test": {
                    "string": "普通字符串",
                    "integer": 42,
                    "float": 3.14159,
                    "boolean": True,
                    "none": None,
                    "numpy_array": np.random.rand(3, 4),
                    "list_of_arrays": [np.random.rand(2, 2) for _ in range(3)],
                    "dict_of_arrays": {
                        "arr1": np.arange(10),
                        "arr2": np.ones(5)
                    }
                },
                "custom_objects": TestObject("混合测试对象", 999, np.random.rand(8, 8))
            }

            print("插入混合类型样本...")
            writer.insert_sample(1, mixed_sample, safe_model=False)

            print(f"操作后数据库大小: {writer.size}")

    except Exception as e:
        print(f"混合操作测试失败: {e}")
        traceback.print_exc()

    # 测试4: 验证修改后的数据
    print("\n4. 验证修改后的数据")
    print("-" * 40)

    try:
        with Reader(dirpath=db_path) as reader:
            print(reader)

            # 验证修改的嵌套样本
            if len(reader) > 0:
                modified_sample = reader[0]
                print("验证修改的嵌套样本:")
                if 'new_nested_field' in modified_sample:
                    print("✓ 新嵌套字段存在")
                    deep_value = modified_sample['new_nested_field']['deeply']['nested']['value']
                    print(f"✓ 深层嵌套值: {deep_value}")

                if 'metadata' in modified_sample and 'user_info' in modified_sample['metadata']:
                    if 'new_preference' in modified_sample['metadata']['user_info']:
                        print("✓ 新偏好设置存在")
                        print(f"  - 字体大小: {modified_sample['metadata']['user_info']['new_preference']['font_size']}")

            # 验证插入的混合样本
            if len(reader) > 1:
                mixed_sample = reader[1]
                print("\n验证混合类型样本:")
                if 'types_test' in mixed_sample:
                    test_data = mixed_sample['types_test']
                    print(f"✓ 包含的数据类型: {list(test_data.keys())}")
                    print(f"✓ 列表数组长度: {len(test_data['list_of_arrays'])}")
                    print(f"✓ 字典数组键: {list(test_data['dict_of_arrays'].keys())}")

                if 'custom_objects' in mixed_sample:
                    obj = mixed_sample['custom_objects']
                    print(f"✓ 自定义对象: {obj}")

    except Exception as e:
        print(f"验证测试失败: {e}")
        traceback.print_exc()

    # 测试5: 边界情况和错误处理
    print("\n5. 边界情况和错误处理测试")
    print("-" * 40)

    try:
        with Reader(dirpath=db_path) as reader:
            # 测试空数据
            empty_meta = reader.get_meta("non_existent_meta")
            print(f"不存在的元数据: {empty_meta}")

            # 测试无效索引
            try:
                invalid_sample = reader[len(reader)]
            except IndexError as e:
                print(f"✓ 无效索引处理: {e}")

            # 测试已删除样本访问
            deleted_keys = reader.deleted_keys
            if deleted_keys:
                physical_key = list(deleted_keys)[0]
                try:
                    deleted_sample = reader.get_delete_sample(physical_key)
                    print(f"✓ 已删除样本访问成功")
                except Exception as e:
                    print(f"✓ 已删除样本访问失败（预期）: {e}")

    except Exception as e:
        print(f"边界测试失败: {e}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)

    # 最终统计和清理建议
    print("\n测试总结:")
    print(f"数据库位置: {os.path.abspath(db_path)}")

    # 显示最终文件大小
    if os.path.exists(db_path):
        size = 0
        if os.path.isdir(db_path):
            for file in os.listdir(db_path):
                file_path = os.path.join(db_path, file)
                if os.path.isfile(file_path):
                    size += os.path.getsize(file_path)
        else:
            size = os.path.getsize(db_path)
        print(f"数据库最终大小: {size / 1024 / 1024:.2f} MB")

    print("\n测试完成，请手动删除测试数据库文件以清理空间")