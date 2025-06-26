"""
sindre.utils3d模块测试用例
测试3D工具类的各种功能
"""

import pytest
import os

# 只测试sindre.utils3d及其子模块的导入和典型API
try:
    import sindre.utils3d
    import sindre.utils3d.mesh
    import sindre.utils3d.algorithm
    import sindre.utils3d.pointcloud_augment
    import sindre.utils3d.dental_tools
    import sindre.utils3d.vedo_tools
    import sindre.utils3d.networks
    UTILS3D_AVAILABLE = True
except ImportError:
    UTILS3D_AVAILABLE = False

@pytest.mark.skipif(not UTILS3D_AVAILABLE, reason="utils3d模块不可用")
def test_utils3d_import():
    import sindre.utils3d
    assert hasattr(sindre.utils3d, "mesh")
    assert hasattr(sindre.utils3d, "algorithm")

def test_mesh_api():
    from sindre.utils3d.mesh import SindreMesh
    mesh = SindreMesh()
    assert hasattr(mesh, "compute_normals")
    assert hasattr(mesh, "show")

def test_algorithm_api():
    from sindre.utils3d.algorithm import labels2colors
    import numpy as np
    labels = np.array([0, 1, 2, 1, 0])
    colors = labels2colors(labels)
    assert colors.shape[0] == 5

def test_pointcloud_augment_api():
    from sindre.utils3d.pointcloud_augment import Flip_np
    import numpy as np
    points = np.random.rand(10, 3)
    flipper = Flip_np(axis_x=True, axis_y=False)
    flipped = flipper(points)
    assert flipped.shape == (10, 3)

def test_networks_import():
    from sindre.utils3d.networks.pointnet2 import pointnet2_ssg
    from sindre.utils3d.networks.dgcnn import DGCNN
    from sindre.utils3d.networks.conv_occ import ConvPointnet
    from sindre.utils3d.networks.point_transformerV3 import PointTransformerV3


if __name__ == "__main__":
    pytest.main([__file__]) 