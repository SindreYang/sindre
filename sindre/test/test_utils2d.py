
import pytest
import os

def test_segment_with_cv2_real_image():
    """测试使用实际图像文件进行分割"""
    from sindre.utils2d.algorithm import segment_with_cv2
    import cv2
    import numpy as np
    # 测试图像路径
    image_path = os.path.join(os.path.dirname(__file__),"data/texture.png")
    
    # 确保图像文件存在
    if not os.path.exists(image_path):
        pytest.skip(f"测试图像不存在: {image_path}")
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        pytest.fail(f"无法读取图像: {image_path}")
    
    # 转换BGR为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    try:
        # 执行分割
        mask = segment_with_cv2(img_rgb)
        
        # 验证返回类型和形状
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.uint8
        assert mask.shape == img_rgb.shape[:2]
        
        # 验证掩码是二值图像
        unique_values = np.unique(mask)
        assert all(v in [0, 255] for v in unique_values), "掩码包含非二值元素"
        
        # 验证掩码包含非零区域（检测到前景）
        foreground_pixels = np.sum(mask == 255)
        assert foreground_pixels > 0, "掩码中没有检测到前景"
        
        # 验证掩码没有覆盖整个图像（保留了背景）
        background_pixels = np.sum(mask == 0)
        assert background_pixels > 0, "掩码覆盖了整个图像，没有保留背景"
        
        # 计算前景占比
        foreground_ratio = foreground_pixels / (mask.shape[0] * mask.shape[1])
        print(f"前景占比: {foreground_ratio:.2%}")
        
    except Exception as e:
        pytest.fail(f"分割过程中出错: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__]) 