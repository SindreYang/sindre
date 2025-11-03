// 导入 Three.js 核心模块和控制器
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// 格式化文件大小
export function formatFileSize(sizeKB) {
    if (sizeKB < 1024) return `${sizeKB} KB`;
    if (sizeKB < 1024 * 1024) return `${(sizeKB / 1024).toFixed(1)} MB`;
    return `${(sizeKB / (1024 * 1024)).toFixed(1)} GB`;
}

// 格式化时间戳
export function formatTimestamp(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
}

// 获取数据库列表
export async function fetchDatabases(filters = {}) {
    try {
        const params = new URLSearchParams();
        if (filters.type) params.append('db_type', filters.type);
        if (filters.tag) params.append('tag', filters.tag);
        if (filters.search) params.append('search', filters.search);

        const response = await fetch(`/api/databases?${params.toString()}`);
        if (!response.ok) throw new Error('获取数据库列表失败');

        const data = await response.json();
        updateDBList(data);
        return data;
    } catch (error) {
        console.error('获取数据库失败:', error);
        document.getElementById('db-list').innerHTML = `
            <div class="text-center text-red-500 py-8">
                <i class="fa fa-exclamation-circle text-2xl mb-2"></i>
                <p>加载数据库失败: ${error.message}</p>
            </div>
        `;
    }
}

// 获取所有标签
export async function fetchTags() {
    try {
        const response = await fetch('/api/tags');
        if (!response.ok) throw new Error('获取标签失败');

        const tags = await response.json();
        const tagFilter = document.getElementById('tag-filter');
        const currentValue = tagFilter.value;

        // 保留第一个"所有标签"选项
        while (tagFilter.options.length > 1) {
            tagFilter.remove(1);
        }

        // 添加标签选项
        tags.forEach(tag => {
            const option = document.createElement('option');
            option.value = tag;
            option.textContent = tag;
            tagFilter.appendChild(option);
        });

        // 恢复选中值
        if (currentValue && tags.includes(currentValue)) {
            tagFilter.value = currentValue;
        }
    } catch (error) {
        console.error('获取标签失败:', error);
    }
}

// 保存数据库信息
export async function saveDatabaseInfo(dbId, dbType, tags) {
    try {
        const response = await fetch(`/api/databases/${dbId}/info`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type: dbType, tags: tags })
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || '保存数据库信息失败');
        }

        return await response.json();
    } catch (error) {
        console.error('保存数据库信息失败:', error);
        throw error;
    }
}

// 更新数据库列表显示
export function updateDBList(databases) {
    const dbListEl = document.getElementById('db-list');
    const dbCountEl = document.getElementById('db-count');

    dbCountEl.textContent = `${databases.length} 个数据库`;

    if (databases.length === 0) {
        dbListEl.innerHTML = `
            <div class="text-center text-gray-500 py-8">
                <i class="fa fa-search text-2xl mb-2"></i>
                <p>未找到匹配的数据库</p>
            </div>
        `;
        return;
    }

    dbListEl.innerHTML = '';
    databases.forEach(db => {
        const dbItem = document.createElement('div');
        dbItem.className = 'p-3 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors flex flex-wrap justify-between items-center';

        // 类型样式
        let typeIcon = 'fa-file-o';
        let typeColor = 'text-gray-500';
        let typeText = db.type || '未设置';

        if (db.type === 'point_cloud') {
            typeIcon = 'fa-cubes';
            typeColor = 'text-accent';
            typeText = '点云';
        } else if (db.type === 'mesh') {
            typeIcon = 'fa-cube';
            typeColor = 'text-accent';
            typeText = '网格';
        } else if (db.type === 'image') {
            typeIcon = 'fa-image';
            typeColor = 'text-secondary';
            typeText = '图像';
        }

        // 数据键和标签信息
        const keysText = db.data_keys?.length
            ? (db.data_keys.length > 3
                ? `${db.data_keys.slice(0, 3).join(', ')}...(共${db.data_keys.length}个)`
                : db.data_keys.join(', '))
            : '无数据键';

        const tagsText = db.tags?.length
            ? (db.tags.length > 2
                ? `${db.tags.slice(0, 2).join(', ')}...`
                : db.tags.join(', '))
            : '无标签';

        // 数据库项HTML（转义JSON避免语法错误）
        const dbJson = JSON.stringify(db).replace(/"/g, '\'');
        dbItem.innerHTML = `
            <div class="flex items-center space-x-2 mb-2 sm:mb-0 w-full sm:w-auto">
                <i class="fa ${typeIcon} ${typeColor}"></i>
                <span class="font-medium">${db.name}</span>
                <span class="text-xs bg-gray-100 px-1.5 py-0.5 rounded">${typeText}</span>
            </div>
            <div class="w-full sm:w-auto text-sm text-gray-500 mb-2 sm:mb-0">数据键: ${keysText}</div>
            <div class="w-full sm:w-auto text-sm text-gray-500 mb-2 sm:mb-0">标签: ${tagsText}</div>
            <div class="w-full sm:w-auto text-sm text-gray-500 mb-2 sm:mb-0">数据量: ${db.length}</div>
            <div class="w-full sm:w-auto flex justify-between items-center">
                <span class="text-xs text-gray-500 mr-4">${formatFileSize(db.size)}</span>
                <div class="flex space-x-2">
                    <button onclick="window.openInfoModal(${dbJson})" class="px-2 py-1 border border-gray-200 text-sm rounded hover:bg-gray-50">
                        <i class="fa fa-info-circle mr-1"></i> 信息
                    </button>
                    <button onclick="window.openEditModal(${dbJson})" class="px-2 py-1 border border-gray-200 text-sm rounded hover:bg-gray-50">
                        <i class="fa fa-edit mr-1"></i> 编辑
                    </button>
                    <button onclick="window.openPreviewModal(${dbJson})" class="px-2 py-1 bg-primary text-white text-sm rounded hover:bg-primary/90">
                        <i class="fa fa-eye mr-1"></i> 预览
                    </button>
                </div>
            </div>
        `;

        dbListEl.appendChild(dbItem);
    });
}

// Three.js 相关状态（使用闭包保存私有状态）
const threeState = {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    light: null,
};

// 初始化Three.js场景
export function initThreeScene(containerId) {
    console.log('开始初始化Three.js场景，容器ID:', containerId);
    // 清除现有场景
    clearThreeScene();

    const container = document.getElementById(containerId);
    if (!container) throw new Error(`Three.js 容器 #${containerId} 不存在`);

    // 若宽度/高度为0，强制设为最小尺寸（避免渲染器初始化失败）用offsetWidth/offsetHeight替代clientWidth/clientHeight
    let width = container.offsetWidth
    let height = container.offsetHeight
    width = width <= 0 ? 300 : width;
    height = height<= 0 ? 300 : height;
    console.log('[调试] 容器最终尺寸（含兜底）:', width, 'x', height);

    try {
        // 创建场景
        threeState.scene = new THREE.Scene();
        threeState.scene.background =new THREE.Color(0xf9fafb);

        // 创建相机
        threeState.camera = new THREE.PerspectiveCamera(
            75,
            width / height,
            0.1,
            1000
        );
        threeState.camera.position.z = 5;

        // 创建渲染器
        threeState.renderer = new THREE.WebGLRenderer({ antialias: true });
        threeState.renderer.setSize(width, height);
        container.appendChild(threeState.renderer.domElement);

        // 添加光源
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        threeState.scene.add(ambientLight);

        threeState.light = new THREE.DirectionalLight(0xffffff, 0.9);
        threeState.light.position.set(5, 10, 7.5);
        threeState.light.castShadow =false;
        threeState.scene.add(threeState.light);

        // 添加控制器
        threeState.controls = new OrbitControls(threeState.camera, threeState.renderer.domElement);


        // 窗口大小变化处理
        const handleResize = () => {
            if (!threeState.renderer) return;
            const newWidth = container.offsetWidth || 800;
            const newHeight = container.offsetHeight || 600; // 同样用offsetHeight
            threeState.renderer.setSize(newWidth, newHeight);
            threeState.camera.aspect = newWidth / newHeight; // 同步更新比例
            threeState.camera.updateProjectionMatrix();
            console.log('[调试] 窗口大小变化，更新渲染器尺寸:', newWidth, 'x', newHeight);
        };


        window.addEventListener('resize', handleResize);
        handleResize();

        // 动画循环
        const animate = () => {
            if (!threeState.renderer) return;
            requestAnimationFrame(animate);

            // 光随相机变化
            threeState.light.position.copy(threeState.camera.position);

            threeState.controls?.update();
            threeState.renderer.render(threeState.scene, threeState.camera);
        };
        animate();

        return Promise.resolve();
    } catch (error) {
        console.error('初始化 Three.js 场景失败:', error);
        container.innerHTML = `
            <div class="text-center text-red-500 py-8">
                <i class="fa fa-exclamation-circle text-2xl mb-2"></i>
                <p>3D 渲染初始化失败: ${error.message}</p>
            </div>
        `;
        throw error;
    }
}

// 渲染点云
export async function renderPointCloud(data) {
    if (!threeState.scene || !data.vertices) throw new Error('点云数据或场景未初始化');

    // 创建点云几何体
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(data.vertices.flat());
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));


    geometry.computeBoundingSphere(); // 计算边界球（包含所有顶点的最小球体）
    const center = geometry.boundingSphere.center; // 模型中心点坐标
    const radius = geometry.boundingSphere.radius; // 模型边界球半径（模型大致"直径"的一半）
    console.log('[调试] 模型边界球 - 中心:', center, '半径:', radius);

    // 设置颜色
    let material;
    if (data.colors) {
        const colors = new Float32Array(data.colors.flat());
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        material = new THREE.PointsMaterial({
            size: 0.1,
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        });
    } else {
        material = new THREE.PointsMaterial({
            size: 0.1,
            color: 0xCC6666,
            transparent: true,
            opacity: 0.8
        });
    }



    // 添加到场景
    const pointCloud = new THREE.Points(geometry, material);
    threeState.scene.add(pointCloud);

    const axesHelper = new THREE.AxesHelper(radius);
    threeState.scene.add(axesHelper);
    console.log('[调试] 已添加坐标轴辅助线');


    // 调整相机：固定在+Z轴，看向模型中心，距离为模型半径的2倍
    if (threeState.camera && threeState.controls) {
        // 相机位置：X和Y与模型中心一致，Z轴在模型中心Z坐标基础上增加2倍半径（确保在+Z方向）
        const cameraZ = center.z + radius * 2; // 距离 = 2倍半径
        threeState.camera.position.set(center.x, center.y, cameraZ);
        console.log('[调试] 相机位置（+Z轴方向）:', threeState.camera.position);

        // 相机看向模型中心（不改变模型位置，直接以模型原始中心为目标）
        threeState.camera.lookAt(center);

        // 轨道控制器目标设置为模型中心（旋转时围绕模型中心）
        threeState.controls.target.copy(center);
        threeState.controls.update(); // 更新控制器状态
        console.log('[调试] 相机已看向模型中心:', center);
    }
    return Promise.resolve();
}

// 渲染网格
export async function renderMesh(data) {
    console.log('开始渲染网格，当前场景状态:', threeState.scene ? '已初始化' : '未初始化');

    // 提取顶点和面数据
    const ori_vertices = data.vertices;
    const ori_faces = data.faces;

    console.log('[调试] 顶点数量:', ori_vertices.length, '面片数量:', ori_faces.length);
    // 创建网格几何体
    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array(ori_vertices.flat());
    const indices = new Uint32Array(ori_faces.flat());
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geometry.setIndex(new THREE.Uint32BufferAttribute(indices, 1));
    geometry.computeVertexNormals();

    geometry.computeBoundingSphere(); // 计算边界球（包含所有顶点的最小球体）
    const center = geometry.boundingSphere.center; // 模型中心点坐标
    const radius = geometry.boundingSphere.radius; // 模型边界球半径（模型大致"直径"的一半）
    console.log('[调试] 模型边界球 - 中心:', center, '半径:', radius);

    // 设置颜色
    let material;
    if (data.colors && data.color_type === 'vertex') {
        const colors = new Float32Array(data.colors.flat());
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        material = new THREE.MeshStandardMaterial({
            vertexColors: true,
            side: THREE.DoubleSide
        });
    } else {
        material = new THREE.MeshStandardMaterial({
            color: 0x994C4C, // 深灰红色
            roughness: 0.7, // 中等粗糙，增强灰度质感
            metalness: 0.2, // 轻微金属感，提升层次
            side: THREE.DoubleSide
        });
    }


    // // 添加立方体（测试)
    // const geometry = new THREE.BoxGeometry(2, 2, 2);
    // const material = new THREE.MeshStandardMaterial({
    //     color: 0x165DFF,
    //     roughness: 0.5,
    //     metalness: 0.3
    // });
    console.log("开始渲染网格")
    // 添加到场景
    const mesh = new THREE.Mesh(geometry, material);
    threeState.scene.add(mesh);
    console.log('[调试] 网格已添加到场景');

    const axesHelper = new THREE.AxesHelper(radius);
    threeState.scene.add(axesHelper);
    console.log('[调试] 已添加坐标轴辅助线');


    // 调整相机：固定在+Z轴，看向模型中心，距离为模型半径的2倍
    if (threeState.camera && threeState.controls) {
        // 相机位置：X和Y与模型中心一致，Z轴在模型中心Z坐标基础上增加2倍半径（确保在+Z方向）
        const cameraZ = center.z + radius * 2; // 距离 = 2倍半径
        threeState.camera.position.set(center.x, center.y, cameraZ);
        console.log('[调试] 相机位置（+Z轴方向）:', threeState.camera.position);

        // 相机看向模型中心（不改变模型位置，直接以模型原始中心为目标）
        threeState.camera.lookAt(center);

        // 轨道控制器目标设置为模型中心（旋转时围绕模型中心）
        threeState.controls.target.copy(center);
        threeState.controls.update(); // 更新控制器状态
        console.log('[调试] 相机已看向模型中心:', center);
    }

    return Promise.resolve();
}

// 清除Three.js场景
export function clearThreeScene() {
    // 移除窗口 resize 监听
    window.removeEventListener('resize', threeState.handleResize);

    // 清理场景资源
    if (threeState.scene) {
        threeState.scene.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(mat => mat.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
        threeState.scene.clear();
    }

    // 移除渲染器DOM
    if (threeState.renderer?.domElement?.parentElement) {
        threeState.renderer.domElement.parentElement.removeChild(threeState.renderer.domElement);
    }

    // 释放控制器
    threeState.controls?.dispose();

    // 重置状态
    threeState.scene = null;
    threeState.camera = null;
    threeState.renderer = null;
    threeState.controls = null;
}

// 加载数据预览（主入口函数）
export async function loadDataPreview(dbPath, dbType, specificDataInfo, previewType, selectedKey, dataIndex) {
    const previewContent = document.getElementById('preview-content');
    previewContent.innerHTML = `
        <div class="text-center text-gray-500 py-8">
            <i class="fa fa-spinner fa-spin text-2xl mb-2"></i>
            <p>加载预览中...</p>
        </div>
    `;

    try {
        // 准备请求参数
        const params = new URLSearchParams();
        params.append('db_path', dbPath);
        params.append('db_type', dbType);
        params.append('preview_type', previewType);
        params.append('selected_key', selectedKey);
        params.append('data_index', dataIndex);

        Object.entries(specificDataInfo).forEach(([key, value]) => {
            params.append(`specific_${key}`, value);
        });

        // 调用预览接口
        const response = await fetch(`/api/preview?${params.toString()}`);
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || '获取预览数据失败');
        }

        const data = await response.json();
        if (data.error) throw new Error(data.error);

        // 渲染预览内容
        if (dbType === 'image') {
            previewContent.innerHTML = `
                <img src="data:image/png;base64,${data.data}" 
                     class="max-w-full max-h-[70vh] object-contain mx-auto my-auto" />
            `;
        } else if (dbType === 'point_cloud' || dbType === 'mesh') {
            previewContent.innerHTML = '<div id="three-container" class="w-full h-[70vh]"></div>';
            // initThreeScene('three-container');
            // dbType === 'point_cloud' ? await renderPointCloud(data) : await renderMesh(data);
            // 等待1帧，确保容器渲染完成
            requestAnimationFrame(() => {
                // 此时容器尺寸已计算，再初始化Three.js
                initThreeScene('three-container');
                // 渲染模型（注意：若renderPointCloud/renderMesh是异步函数，需用await）
                if (dbType === 'point_cloud') {
                    renderPointCloud(data);
                } else {
                    renderMesh(data);
                }
            });
        }

        // 修改后：

    } catch (error) {
        console.error('加载预览失败:', error);
        previewContent.innerHTML = `
            <div class="text-center text-red-500 py-8">
                <i class="fa fa-exclamation-circle text-2xl mb-2"></i>
                <p>${error.message}</p>
            </div>
        `;
    }
}

// 暴露所有函数到全局（供非模块脚本使用）
window.tools = {
    formatFileSize,
    formatTimestamp,
    fetchDatabases,
    fetchTags,
    saveDatabaseInfo,
    updateDBList,
    initThreeScene,
    renderPointCloud,
    renderMesh,
    clearThreeScene,
    loadDataPreview
};