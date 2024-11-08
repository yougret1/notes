import numpy as np


def bilinear_interpolation(image, out_height, out_width, corner_align = False):
    # 获取输入图象的宽高
    height, width = image.shape[:2] #image.shape[:3] 表示取彩色图片的高、宽、通道

    # 创建输出图象
    output_image = np.zeros((out_height, out_width), dtype=np.float32) #创建一个 h * w 的矩阵

    # 计算x,y轴缩放因子
    scale_x_corner = float(width - 1) / (out_width - 1)
    scale_y_corner = float(height - 1) / (out_height - 1)

    scale_x = float(width) / out_width
    scale_y = float(height) / out_height

    # 便利输出图象的每个像素，分别计算其在输出图象中最近的四个像素的坐标，然后按照加权值计算当前像素的像素点
    for out_y in range(out_height):
        for out_x in range(out_width):
            if corner_align == True:
                # 计算当前像素在输入图象中的位置
                x = out_x * scale_x_corner
                y = out_y * scale_y_corner
            else:
                x = (out_x + 0.5) * scale_x - 0.5
                y = (out_y + 0.5) * scale_y - 0.5
                x = np.clip(x, 0, width - 1)
                y = np.clip(y, 0, height - 1)
                # np.clip(a, a_min, a_max, out=None)  ## 是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值,将数组限制在最小值和最大值之间
                # a：输入矩阵；
                # a_min：被限定的最小值，所有比a_min小的数都会强制变为a_min；
                # a_max：被限定的最大值，所有比a_max大的数都会强制变为a_max；
                # out：可以指定输出矩阵的对象，shape与a相同