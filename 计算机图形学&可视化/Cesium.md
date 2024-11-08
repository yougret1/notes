

[参考文档](https://cesium.com/learn/cesiumjs-learn/cesiumjs-quickstart/#step-2-set-up-the-cesiumjs-client)

# 初始化要求

1. public的文件夹中需要放入node_modules中下载过来的cesium中的四个包，Assets,ThirdParty,Widgets,Workers
2. 全局css需引入，import "./Widgets/widgets.css"
3. window.CESIUM_BASE_URL = "/cesium/"; 这里放置public这四个包的路径 
   `The URL on your server where CesiumJS's static files are hosted.`

<u>注意！所有的cesium相关的url如果是调用了内置api，那么目前的路径就是cesium的路径，而不是当前文件路径</u>

## 自定义准备

设置token , [token获取地址](https://ion.cesium.com/tokens?page=1)

# 处理

#### 设置经纬度

中国为例

```js
Cesium.Camera.DEFAULT_VIEW_RECTANGLE = Cesium.Rectangle.fromDegrees(
  89.5,20.4,110.4,61.2
)
```



# 使用其他来源地图data

[样例](https://blog.csdn.net/hongxianqiang/article/details/140527093)

```js
  let ter = addTMap(viewer, "ter");
  let vec = addTMap(viewer, "vec");
  vec.alpha = 0.5;
  function addTMap(viewer, layer) {
    // 添加天地图影像注记底图
    const tMapImagery = new Cesium.WebMapTileServiceImageryProvider({
      url: `http://t0.tianditu.gov.cn/${layer}_w/wmts?tk=65f8beacb793d1f12803cd752c96185d`,
      layer,
      style: "default",
      tileMatrixSetID: "w",
      format: "tiles",
      maximumLevel: 18,
    });
    let tem = viewer.imageryLayers.addImageryProvider(tMapImagery);
    return tem;
  }
```

# 地理空间数据云使用

使用cesiumlab3来对下载的地理信息进行处理

地形切片 -> 处理参数 三角算法vcg -> 输出文件 存储类型散列 -> 提交处理

# ERROR handle

> about:blank:1 Blocked script execution in 'about:blank' because the document's frame is sandboxed and the 'allow-scripts' permission is not set.

提示js没有权限调用iframe，

```js
  var viewer = new Cesium.Viewer("cesiumContainer",{
    infoBox: false //解决办法
  });
```

