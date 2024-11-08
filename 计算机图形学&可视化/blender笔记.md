# blender操作

## 基本操作

控制观察视角：鼠标中键

平移视图：Shift+鼠标中间

缩放视图：滚动鼠标中间滚轮

删除：左边选择select box，右边选择相应图层，DEL键，删除

新建物体：shift+A

跟随移动/缩放/旋转：选中想要跟随移动的物体，+ R / S / G，后跟随移动，鼠标右击解除

- 跟随移动中按下x，y，z键位，那么就会跟着x/y/z轴进行相对应
- ALT + R / S  / G 撤销移动/缩放/旋转，使其回到初始位置
- tip：rotate / scale / glide

显示：

- 隐藏物体：选中后，H
- 显示全部隐藏物体 ALT + H
- 把没有被选中的物体隐藏 shift + H
- tip：hide

移动复制：shift + D

刷选工具：C ，ESC退出刷选

快捷切换上下左右视图，~ 键，或者1(Y轴)2向下3(X轴)4(向左)5(平面)6(向右)7(Z轴)8(向上)9(反方向) 0(摄像机视图) 。(快速预览选中东西)



### 基础视图



# google相关

## 地图区块建模/城市，山体建模

前提，需要[blenderGIS-228](F:\blender\PlugBackUp)插件，Add-on 中显示 `3D View:BlenderGIS`

以及 `Node:Node:Wrangler` 插件开启

### 山体

GIS(左上) -> Web geodata -> BaseMap 获取基本信息

1. G键，弹出定位信息对话框，显示级别越高精度越高范围越小，
2. 确定范围后按下 E键 固定趋于，贴图映射至平面

GIS(左上) -> Web geodata -> Get elevation(SRTM) 获取高度图

1. 调整高度：右下框 Modifiiters :wrench: (蓝色小钳子上数下10个) ->强度力度提高
2. 进编辑模式 -> A键权限 ->右键细分 ->左下角弹出后打开选择切分次数 ->推出编辑模式
3. 不必要，若想出现梯田状。坐下最后texture ->采样-> 勾选插值类型
4. 进入Modifiiters ->应用修改器 (下拉框中)
5. 增加厚度 -> 编辑模式下(tap) ->A键，后E键，再Z键，鼠标往下移动 -> S键后E键，后小键盘0键，回车

城市

GIS(左上) -> Web geodata -> BaseMap 获取基本信息(Source :OSM)

1. G键，弹出定位信息对话框，显示级别越高精度越高范围越小，
2. 确定范围后按下 E键 固定趋于，贴图映射至平面

GIS(左上) -> Web geodata -> Get OSM shift多选

