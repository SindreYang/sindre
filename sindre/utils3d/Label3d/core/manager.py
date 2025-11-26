import json
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtWidgets import  QWidget


class CoreSignals(QObject):
    """信号集合（统一管理，便于扩展）"""
    # 信号1：提示信息（参数：文本str)
    signal_info = pyqtSignal(str)
    # 信号2: 结束信号(参数：bool)
    signal_close = pyqtSignal(bool)
    # 信号3: dock组件替换
    signal_dock = pyqtSignal(QWidget)
    # 信号4: 当前处理数据
    signal_data = pyqtSignal(dict)
    # 信号5: labels更新
    signal_labels_updated = pyqtSignal(dict)
    # 信号5: labels选择信号
    signal_labels_clicked = pyqtSignal(str) #当前选择标签
    
class LabelManager:
    """标签管理器"""
    def __init__(self):
        # 初始化一个空的标签字典
        # 结构为: { '标签名': {'color': (R, G, B), 'used': Boolean}, ... }
        self.labels = {}
        # 当前选中的标签名，初始为 None
        self.current_label = None

    
    def get_label_names(self):
        """获取所有标签名称"""
        return list(self.labels.keys())
    
    def get_label_color(self, label_name):
        """根据标签名获取标签颜色"""
        if label_name in self.labels:
            return self.labels[label_name]['color']
        # 如果找不到，返回一个默认颜色
        return (255, 0, 0)
    
    
    def add_label(self, name, color):
        """
        添加新标签。
        :param name: 标签名称 (将作为字典的键)
        :param color: 标签颜色，一个三元组 (R, G, B)
        :return: 如果添加成功则返回 True，如果标签名已存在则返回 False
        """
        if name not in self.labels:
            self.labels[name] = {'color': color, 'used': False}
            # 如果是第一个标签，可以自动将其设为当前标签
            if self.current_label is None:
                self.current_label = name
            return True
        return False
    
    def remove_label(self, name):
        """
        根据标签名删除标签。
        :param name: 要删除的标签名称
        :return: 如果删除成功则返回 True，否则返回 False
        """
        if name in self.labels:
            del self.labels[name]
            # 如果删除的是当前标签，则将 current_label 置为 None 或第一个剩余标签
            if self.current_label == name:
                self.current_label = next(iter(self.labels.keys())) if self.labels else None
            return True
        return False
    
    def update_label(self, old_name, new_name=None, new_color=None):
        """
        更新标签的名称或颜色。
        :param old_name: 原始标签名称
        :param new_name: 新的标签名称 (可选)
        :param new_color: 新的颜色 (可选)
        :return: 如果更新成功则返回 True，否则返回 False
        """
        if old_name not in self.labels:
            return False

        # 更新颜色
        if new_color is not None:
            self.labels[old_name]['color'] = new_color

        # 更新名称 (需要处理字典键的变更)
        if new_name is not None and new_name != old_name:
            if new_name not in self.labels:
                # 将旧键的值赋给新键
                self.labels[new_name] = self.labels.pop(old_name)
                # 如果更新的是当前标签的名称，同步更新 current_label
                if self.current_label == old_name:
                    self.current_label = new_name
            else:
                # 新名称已存在，更新失败
                return False

        return True
    
    def mark_label_used(self, label_name):
        """
        标记标签为已使用。
        :param label_name: 要标记的标签名称
        :return: 如果标记成功则返回 True，否则返回 False
        """
        if label_name in self.labels:
            self.labels[label_name]['used'] = True
            return True
        return False
    
    def mark_label_unused(self, label_name):
        """
        标记标签为未使用。
        :param label_name: 要标记的标签名称
        :return: 如果标记成功则返回 True，否则返回 False
        """
        if label_name in self.labels:
            self.labels[label_name]['used'] = False
            return True
        return False
    def is_label_used(self, label_name):
        """
        检查标签是否已被使用。
        :param label_name: 要检查的标签名称
        :return: 如果标签存在且已使用则返回 True，否则返回 False
        """
        if label_name in self.labels:
            return self.labels[label_name]['used']
        return False

    
    def get_used_labels(self):
        """获取所有已使用的标签名称列表"""
        return [name for name, info in self.labels.items() if info['used']]
    def get_unused_labels(self):
        """获取所有未使用的标签名称列表"""
        return [name for name, info in self.labels.items() if not info['used']]
    def reset_all_labels(self):
        """重置所有标签为未使用状态"""
        for info in self.labels.values():
            info['used'] = False
   
    def set_current_label(self, label_name):
        """
        设置当前活跃的标签。
        :param label_name: 要设置为当前的标签名称
        :return: 如果设置成功则返回 True，否则返回 False
        """
        if label_name in self.labels:
            self.current_label = label_name
            return True
        return False
    def clear_all_labels(self):
        """清空所有标签"""
        self.labels.clear()
        self.current_label = None
