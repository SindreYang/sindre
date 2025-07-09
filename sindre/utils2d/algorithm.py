import numpy as np
from sindre.general.logs import CustomLogger
log = CustomLogger(logger_name="algorithm2d").get_logger()



def detect_ellipses_with_cv2(img_raw):
    """对输入图像进行椭圆检测
    
    该函数执行以下主要步骤：
    1. 图像预处理（灰度转换、滤波、边缘增强）
    2. 自适应阈值处理
    3. 使用EdgeDrawing算法检测椭圆
    4. 可视化检测结果并返回
    
    Args:
        img_raw (numpy.ndarray): 输入BGR彩色图像
        
    Returns:
        tuple: (包含椭圆标注的结果图像, 椭圆关键点列表, 椭圆像素坐标列表)
    """
    # === 图像预处理 ===
    # 转换为灰度图像
    gray = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
    
    # 高斯模糊降噪 (3x3核，标准差=5)
    scale_blur1 = cv2.GaussianBlur(gray, (3, 3), 5)
    
    # 使用双边滤波保留边缘 (直径=5，颜色空间标准差=9，坐标空间标准差=9)
    scale_patch = cv2.bilateralFilter(scale_blur1, 5, 9, 9)
    
    # 二次高斯模糊处理 (自动核大小，标准差=11)
    scale_blur = cv2.GaussianBlur(scale_patch, (0, 0), 11)
    
    # 非锐化掩蔽增强边缘 (1.5 * scale_patch - 0.5 * scale_blur)
    scale_patch = cv2.addWeighted(scale_patch, 1.5, scale_blur, -0.5, 0.0)
    gray = scale_patch
    
    # === 自适应阈值处理 ===
    # 使用均值自适应阈值 (邻域大小21x21，常数偏移31，二值化类型为反二进制)
    img_bin = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY_INV, 
        21, 31
    )
    img = np.array(img_bin, dtype='uint8')
    
    # === 椭圆检测 ===
    # 创建EdgeDrawing对象并配置参数
    ed = cv2.ximgproc.createEdgeDrawing()
    EDParams = cv2.ximgproc_EdgeDrawing_Params()
    
    # 参数配置说明：
    EDParams.MinPathLength = 2      # 最小边缘路径长度 (降低可检测更短边缘)
    EDParams.PFmode = False          # 是否使用渐进式处理
    EDParams.MinLineLength = 3       # 最小线段长度
    EDParams.NFAValidation = True    # 启用NFA验证
    EDParams.GradientThresholdValue = 1  # 梯度阈值 (提高可保留弱边缘)
    EDParams.AnchorThresholdValue = 5    # 锚点阈值 (值越小锚点越多)
    EDParams.MaxDistanceBetweenTwoLines = 5  # 合并线段的最大距离
    EDParams.EdgeDetectionOperator = cv2.ximgproc.EDGE_DRAWING_SCHARR  # Scharr边缘检测算子
    EDParams.ScanInterval = 1        # 像素扫描间隔
    EDParams.Sigma = 1               # 高斯模糊系数
    ed.setParams(EDParams)
    
    # 执行边缘检测
    ed.detectEdges(img)
    
    # === 结果处理与可视化 ===
    result_img = img_raw.copy()  # 创建结果图像副本
    key_points = []              # 存储椭圆关键点
    ellipse_pixels_list = []     # 存储椭圆像素坐标
    
    # 检测椭圆
    ellipses = ed.detectEllipses()
    
    if ellipses is not None:
        # 为椭圆掩码创建空白画布 (使用灰度图尺寸)
        height, width = gray.shape[:2]
        
        for ellipse in ellipses:
            # 解析椭圆参数
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            axis1 = int(ellipse[0][2]) + int(ellipse[0][3])
            axis2 = int(ellipse[0][2]) + int(ellipse[0][4])
            axes = (axis1, axis2)
            angle = ellipse[0][5]
            
            # 计算关键点尺寸 (平均半径)
            radius_avg = ellipse[0][2] + (ellipse[0][3] + ellipse[0][4]) / 2.0
            
            # 创建关键点对象 (位置, 尺寸, 角度)
            kpt = cv2.KeyPoint(ellipse[0][0], ellipse[0][1], radius_avg, angle)
            key_points.append(kpt)
            
            # 根据椭圆有效性设置颜色 (绿色有效, 黄色可疑)
            color = (0, 255, 0)  # BGR绿色
            if ellipse[0][2] == 0:  # 短轴为0表示可疑椭圆
                color = (0, 255, 255)  # BGR黄色
            
            # 在结果图像上绘制椭圆
            cv2.ellipse(result_img, center, axes, angle, 0, 360, color, 1, cv2.LINE_AA)
            
            # 创建椭圆掩码并提取像素坐标
            ellipse_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(ellipse_mask, center, axes, angle, 0, 360, 255, 1, cv2.LINE_AA)
            ellipse_pixels = np.argwhere(ellipse_mask > 0)
            ellipse_pixels_list.append(ellipse_pixels[:, [1, 0]])  # 转换为(x,y)格式
    
    # 保存结果图像
    #cv2.imwrite("ellipse_detection_result.jpg", result_img)
    log.info('椭圆检测完成')
    
    return result_img, key_points, ellipse_pixels_list



def segment_with_cv2(img: np.ndarray):
    """
    使用OpenCV对图像进行分割，提取主要内容区域
    
    Args:
        img: 输入的RGB图像数组
    
    Returns:
        mask: 二值掩码，主要内容区域为白色(255)，背景为黑色(0)


    Note:
        img = cv2.imread(image_path)
        # 转换BGR为RGB (OpenCV默认以BGR模式读取)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 执行图像分割
        mask = segment_with_cv2(img_rgb)
    """
    import cv2
    from scipy import ndimage
    from skimage import measure
    # 阈值处理
    img = np.min(img, axis=2)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 8)
    img = cv2.bitwise_not(img)

    # 形态学操作：闭运算填充小孔洞，膨胀连接相邻区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=2)

    # 洪水填充算法：从图像边缘开始填充背景区域
    mask = np.zeros([img.shape[0]+2, img.shape[1]+2], np.uint8)
    mask[1:-1, 1:-1] = img.copy()

    # im_floodfill 是漫水填充的结果，开始时全是白色
    im_floodfill = np.full(img.shape, 255, np.uint8)

    # 在每个图像边缘选择 10 个点。将其用作漫水填充的种子点
    h, w = img.shape[:2]
    for x in range(0, w-1, 10):
        cv2.floodFill(im_floodfill, mask, (x, 0), 0)
        cv2.floodFill(im_floodfill, mask, (x, h-1), 0)
    for y in range(0, h-1, 10):
        cv2.floodFill(im_floodfill, mask, (0, y), 0)
        cv2.floodFill(im_floodfill, mask, (w-1, y), 0)

    # 确保边缘不是字符。这对于轮廓查找是必要的。
    im_floodfill[0, :] = 0
    im_floodfill[-1, :] = 0
    im_floodfill[:, 0] = 0
    im_floodfill[:, -1] = 0

    # 寻找最大轮廓并创建掩码
    mask2 = cv2.bitwise_not(im_floodfill)
    mask = None
    biggest = 0

    contours = measure.find_contours(mask2, 0.0)
    for c in contours:
        # 创建当前轮廓的掩码
        x = np.zeros(mask2.T.shape, np.uint8)
        cv2.fillPoly(x, [np.int32(c)], 1)
        size = len(np.where(x == 1)[0])
        # 计算轮廓面积
        if size > biggest:
            mask = x
            biggest = size

    if mask is None:
        msg = '未在图像中找到有效轮廓'
        log.critical(msg)
        assert False, msg

    mask = ndimage.binary_fill_holes(mask).astype(int)
    mask = 255 * mask.astype(np.uint8)

    return mask.T

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('WebAgg')
    import cv2
    image_path = "/home/up3d/sindre/sindre/test/data/texture.png"
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        
        # 转换BGR为RGB (OpenCV默认以BGR模式读取)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 执行图像分割
        mask = detect_ellipses_with_cv2(img_rgb)
        
    
    except Exception as e:
        print(f"处理图像时出错: {e}")
