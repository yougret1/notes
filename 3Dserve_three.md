# three

# error

#### `glsl`被识别成js的问题

[webpack中文官网](https://cli.vuejs.org/zh/config/#configurewebpack)

由于`webpack`导致的，具体配置如下

`vue.config.js`中配置如下，`webpack`方法式声明

```
const path = require('path')

module.exports = {
  chainWebpack: config => {
    config.module
      .rule('glsl')
      .test(/\.glsl$/)
      .use('webpack-glsl-loader')
      .loader('webpack-glsl-loader')
      .end()
  }
}

```

# 其他

## google地图区块建模/城市，山体建模

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