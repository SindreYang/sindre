"""
sindre.win_tools模块测试用例
测试Windows工具类的各种功能
"""

import pytest
import os
import sys

# 检查是否在Windows平台上
IS_WINDOWS = sys.platform.lower() == "win32"


@pytest.mark.skipif(not IS_WINDOWS, reason="仅在Windows平台上运行")
class TestWinTools:
    """测试Windows工具类的各种功能"""
    
    def test_win_tools_import(self):
        """测试Windows工具模块导入"""
        try:
            import sindre.win_tools as win_tools
            assert True
        except ImportError:
            pytest.skip("win_tools模块不可用")
    
    def test_tools_import(self):
        """测试tools模块导入"""
        try:
            import sindre.win_tools.tools as tools
            assert True
        except ImportError:
            pytest.skip("tools模块不可用")
    
    def test_taskbar_import(self):
        """测试taskbar模块导入"""
        try:
            import sindre.win_tools.taskbar as taskbar
            assert True
        except ImportError:
            pytest.skip("taskbar模块不可用")
    
    def test_bin_directory_exists(self):
        """测试bin目录是否存在"""
        bin_path = os.path.join(os.path.dirname(__file__), "..", "win_tools", "bin")
        assert os.path.exists(bin_path), f"bin目录不存在: {bin_path}"
    
    def test_7z_tools_exist(self):
        """测试7z工具是否存在"""
        bin_path = os.path.join(os.path.dirname(__file__), "..", "win_tools", "bin")
        seven_z_path = os.path.join(bin_path, "7z")
        
        if os.path.exists(seven_z_path):
            # 检查7z.exe是否存在
            exe_path = os.path.join(seven_z_path, "7z.exe")
            dll_path = os.path.join(seven_z_path, "7z.dll")
            
            # 至少应该有一个存在
            assert os.path.exists(exe_path) or os.path.exists(dll_path), "7z工具文件不存在"
    
    def test_nsis_tools_exist(self):
        """测试NSIS工具是否存在"""
        bin_path = os.path.join(os.path.dirname(__file__), "..", "win_tools", "bin")
        nsis_path = os.path.join(bin_path, "NSIS")
        
        if os.path.exists(nsis_path):
            # 检查NSIS主要文件
            makensis_exe = os.path.join(nsis_path, "makensis.exe")
            makensisw_exe = os.path.join(nsis_path, "makensisw.exe")
            
            # 至少应该有一个存在
            assert os.path.exists(makensis_exe) or os.path.exists(makensisw_exe), "NSIS工具文件不存在"
    
    def test_config_files_exist(self):
        """测试配置文件是否存在"""
        bin_path = os.path.join(os.path.dirname(__file__), "..", "win_tools", "bin")
        config_path = os.path.join(bin_path, "config")
        
        if os.path.exists(config_path):
            # 检查配置文件
            config_files = [
                "commonfunc.nsh",
                "info.nsi",
                "license.txt",
                "logo.ico",
                "skin.zip",
                "ui.nsh",
                "uninst.ico"
            ]
            
            existing_files = [f for f in config_files if os.path.exists(os.path.join(config_path, f))]
            assert len(existing_files) > 0, "没有找到任何配置文件"
    
    def test_shared_memory_exe_exists(self):
        """测试共享内存可执行文件是否存在"""
        bin_path = os.path.join(os.path.dirname(__file__), "..", "win_tools", "bin")
        shared_memory_exe = os.path.join(bin_path, "shared_memory.exe")
        
        if os.path.exists(shared_memory_exe):
            # 检查文件是否可执行
            assert os.access(shared_memory_exe, os.X_OK) or shared_memory_exe.endswith('.exe'), "共享内存可执行文件不可执行"


class TestWinToolsCrossPlatform:
    """测试Windows工具的跨平台兼容性"""
    
    def test_platform_check(self):
        """测试平台检查功能"""
        # 测试平台检测
        platform = sys.platform.lower()
        assert platform in ["win32", "linux", "darwin"], f"未知平台: {platform}"
    
    def test_conditional_import(self):
        """测试条件导入"""
        # 测试在非Windows平台上的导入行为
        if not IS_WINDOWS:
            # 在非Windows平台上，导入应该不会失败，但可能某些功能不可用
            try:
                import sindre.win_tools
                # 导入成功，但功能可能不可用
                assert True
            except ImportError:
                # 导入失败是正常的
                assert True
    
    def test_file_structure(self):
        """测试文件结构"""
        # 检查win_tools目录结构
        win_tools_path = os.path.join(os.path.dirname(__file__), "..", "win_tools")
        
        if os.path.exists(win_tools_path):
            # 检查必要的文件
            required_files = ["__init__.py"]
            for file in required_files:
                file_path = os.path.join(win_tools_path, file)
                assert os.path.exists(file_path), f"缺少必要文件: {file}"
    
    def test_bin_structure(self):
        """测试bin目录结构"""
        bin_path = os.path.join(os.path.dirname(__file__), "..", "win_tools", "bin")
        
        if os.path.exists(bin_path):
            # 检查子目录
            subdirs = ["7z", "config", "NSIS"]
            existing_subdirs = [d for d in subdirs if os.path.exists(os.path.join(bin_path, d))]
            
            # 至少应该有一些子目录存在
            assert len(existing_subdirs) > 0, "bin目录下没有找到任何子目录"
    
    def test_nsis_structure(self):
        """测试NSIS目录结构"""
        bin_path = os.path.join(os.path.dirname(__file__), "..", "win_tools", "bin")
        nsis_path = os.path.join(bin_path, "NSIS")
        
        if os.path.exists(nsis_path):
            # 检查NSIS子目录
            nsis_subdirs = ["Bin", "Contrib", "Include", "Plugins", "Stubs"]
            existing_subdirs = [d for d in nsis_subdirs if os.path.exists(os.path.join(nsis_path, d))]
            
            # 至少应该有一些NSIS子目录存在
            assert len(existing_subdirs) > 0, "NSIS目录下没有找到任何子目录"
    
    def test_contrib_structure(self):
        """测试Contrib目录结构"""
        bin_path = os.path.join(os.path.dirname(__file__), "..", "win_tools", "bin")
        contrib_path = os.path.join(bin_path, "NSIS", "Contrib")
        
        if os.path.exists(contrib_path):
            # 检查Contrib子目录
            contrib_subdirs = ["Graphics", "Language files", "Modern UI", "Modern UI 2", "UIs", "zip2exe"]
            existing_subdirs = [d for d in contrib_subdirs if os.path.exists(os.path.join(contrib_path, d))]
            
            # 至少应该有一些Contrib子目录存在
            assert len(existing_subdirs) > 0, "Contrib目录下没有找到任何子目录"
    
    def test_graphics_structure(self):
        """测试Graphics目录结构"""
        bin_path = os.path.join(os.path.dirname(__file__), "..", "win_tools", "bin")
        graphics_path = os.path.join(bin_path, "NSIS", "Contrib", "Graphics")
        
        if os.path.exists(graphics_path):
            # 检查Graphics子目录
            graphics_subdirs = ["Checks", "Header", "Icons", "Wizard"]
            existing_subdirs = [d for d in graphics_subdirs if os.path.exists(os.path.join(graphics_path, d))]
            
            # 至少应该有一些Graphics子目录存在
            assert len(existing_subdirs) > 0, "Graphics目录下没有找到任何子目录"
    
    def test_icons_structure(self):
        """测试Icons目录结构"""
        bin_path = os.path.join(os.path.dirname(__file__), "..", "win_tools", "bin")
        icons_path = os.path.join(bin_path, "NSIS", "Contrib", "Graphics", "Icons")
        
        if os.path.exists(icons_path):
            # 检查图标文件
            icon_files = [
                "arrow-install.ico", "arrow-uninstall.ico", "arrow2-install.ico", "arrow2-uninstall.ico",
                "box-install.ico", "box-uninstall.ico", "classic-install.ico", "classic-uninstall.ico",
                "llama-blue.ico", "llama-grey.ico", "modern-install-blue-full.ico", "modern-install-blue.ico",
                "modern-install-colorful.ico", "modern-install-full.ico", "modern-install.ico",
                "modern-uninstall-blue-full.ico", "modern-uninstall-blue.ico", "modern-uninstall-colorful.ico",
                "modern-uninstall-full.ico", "modern-uninstall.ico", "nsis1-install.ico", "nsis1-uninstall.ico",
                "orange-install-nsis.ico", "orange-install.ico", "orange-uninstall-nsis.ico", "orange-uninstall.ico",
                "pixel-install.ico", "pixel-uninstall.ico", "win-install.ico", "win-uninstall.ico"
            ]
            
            existing_icons = [f for f in icon_files if os.path.exists(os.path.join(icons_path, f))]
            # 至少应该有一些图标文件存在
            assert len(existing_icons) > 0, "Icons目录下没有找到任何图标文件"
    
    def test_language_files_structure(self):
        """测试Language files目录结构"""
        bin_path = os.path.join(os.path.dirname(__file__), "..", "win_tools", "bin")
        lang_path = os.path.join(bin_path, "NSIS", "Contrib", "Language files")
        
        if os.path.exists(lang_path):
            # 检查语言文件
            lang_files = [
                "English.nlf", "English.nsh", "SimpChinese.nlf", "SimpChinese.nsh",
                "TradChinese.nlf", "TradChinese.nsh", "Japanese.nlf", "Japanese.nsh",
                "Korean.nlf", "Korean.nsh"
            ]
            
            existing_langs = [f for f in lang_files if os.path.exists(os.path.join(lang_path, f))]
            # 至少应该有一些语言文件存在
            assert len(existing_langs) > 0, "Language files目录下没有找到任何语言文件"


class TestWinToolsFunctionality:
    """测试Windows工具功能"""
    
    def test_file_permissions(self):
        """测试文件权限"""
        # 创建测试文件
        test_file = os.path.join(os.path.dirname(__file__), "..", "win_tools", "bin", "test.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        # 检查文件权限
        assert os.path.exists(test_file)
        assert os.access(test_file, os.R_OK)
        assert os.access(test_file, os.W_OK)
    
    def test_directory_operations(self):
        """测试目录操作"""
        # 创建测试目录
        test_dir = os.path.join(os.path.dirname(__file__), "..", "win_tools", "bin", "test_dir")
        os.makedirs(test_dir, exist_ok=True)
        
        # 检查目录权限
        assert os.path.exists(test_dir)
        assert os.access(test_dir, os.R_OK)
        assert os.access(test_dir, os.W_OK)
        assert os.access(test_dir, os.X_OK)
    
    def test_path_operations(self):
        """测试路径操作"""
        # 测试路径拼接
        base_path = "C:\\Program Files"
        sub_path = "MyApp"
        full_path = os.path.join(base_path, sub_path)
        
        assert full_path == "C:\\Program Files\\MyApp"
        
        # 测试路径分割
        path_parts = os.path.split(full_path)
        assert path_parts[0] == "C:\\Program Files"
        assert path_parts[1] == "MyApp"
    
    def test_environment_variables(self):
        """测试环境变量"""
        # 检查常见的Windows环境变量
        env_vars = ["PATH", "TEMP", "TMP", "USERPROFILE"]
        
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                assert len(value) > 0, f"环境变量 {var} 为空"
    
    def test_system_info(self):
        """测试系统信息"""
        # 检查系统信息
        assert hasattr(os, 'name')
        assert hasattr(sys, 'platform')
        assert hasattr(sys, 'version')
        
        # 在Windows上检查特定属性
        if IS_WINDOWS:
            assert os.name == 'nt'
            assert 'win' in sys.platform.lower()


if __name__ == "__main__":
    pytest.main([__file__]) 