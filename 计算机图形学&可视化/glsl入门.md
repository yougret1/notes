所有像素点**<u>并行运算</u>**



GLSL 是强语言,面向对象，可以实现继承和多态

点乘可以用来计算一个向量在另一个向量方向上的投影长度，叉乘（也称为外积）的几何意义则是，它可以用来计算两个向量所确定的平行四边形的面积，并且方向垂直于这个平行四边形所在的平面。叉乘可以用来描述两个向量所确定的平行四边形的性质，包括面积和方向。

## vec()向量定义

vector 向量

```glsl
vec2 v = vec2(0.5);
w = v * 2.0;
// 意味着 v.x = 0.5,v.y = 0.5
// 意味着 w.x = 1,w.y = 1

vec3 v = vec3(0.5);
// 意味着 v.x = 0.5,v.y = 0.5,v.z = 0.5 
```

## Integers/Booleans基本类型定义

整数/布尔值

```glsl
// 整数不同维度向量
ivec2 i2 = ivec2(725);
ivec3 i3 = ivec3(725);
ivec4 i4 = ivec4(725);
//浮点数不同维度向量
bvec2 b2 = bvec2(true);
bvec3 b3 = bvec3(false);
bvec4 b4 = bvec4(true);
```

## Loop基本循环

in GLSL

```glsl
const int count = 10;
for(let i = 0 ;i < count ; i++){
    //
}
```

in JavaScript

```glsl
let count = 10;
for(let i = 0 ;i < count ; i++){
    //
}
```

## function

可实现多态

in GLSL

```glsl
bool fun(vec2 pt , vec4 rec){
    bool result = false;
    //do something
    return result;
}
// 多态
bool fun(vec2 pt , vec4 rec , vec2 re){
    return true;
}
```

## Vertex

注意：所有的vertex格式都是竖着的，e.g. 1*3：

   1
{  2  }
   3

vertex相加

```glsl
uniform vec3 u_color;
vec3 a = vec3(0.3,0.5,0.3);

void main() {	
    gl_FragColor = vec4(u_color + a, 0.1);
}
```



## Matrix

`modelMatrix`：模型矩阵用于描述一个物体相对于世界坐标系的位置、旋转和缩放。当一个物体被绘制时，它的顶点坐标会先经过模型矩阵的变换，将其从局部坐标系（物体自身的坐标系）转换到世界坐标系（全局坐标系）。

`viewMatrix`：视图矩阵（或观察矩阵）描述了观察者的位置和方向，以及观察者的视角。视图矩阵通常用于将场景中的物体从世界坐标系转换到观察者的坐标系（也称为摄像机坐标系）。

`projectionMatrix`：投影矩阵用于将三维空间中的坐标转换为二维屏幕坐标，实现透视效果和远近物体的大小变化。投影矩阵通常描述了观察者的视锥体（视景体），并将其中的物体投影到一个二维平面上。

`modelViewMatrix`：模型-视图矩阵是模型矩阵和视图矩阵的组合，它将物体从局部坐标系变换到观察者的坐标系。在图形渲染中，通常会将模型矩阵和视图矩阵相乘得到模型-视图矩阵，然后再将投影矩阵应用于模型-视图矩阵，最终得到物体在屏幕空间的位置。

Matrix定义

```glsl
float s = sin(time),c = cos(time);
mat2(c,-s,s,c); //二阶矩阵，里面数值用于二阶点旋转形变
```



## Uniforms

uniforms将从数据从控制程序传递到着色器

每一个uniform将会存储一个正常的值，作用与每一个点(vertex)和像素(pixel)

u_mouse:存储鼠标此时的x，y的信息

![uniform.屏幕鼠标坐标](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\uniform.屏幕鼠标坐标.png)

u_mouse 存储鼠标坐标

u_resolution 存储窗口的整体像素尺寸(浏览器大小)、

`u_mouse.x = u_mouse[0]`



## functions

### `gl_FragCoord`

vec4 type(x,y,z,w)

x & y eg. `gl_FragColor.xy`

![gl_FragCoord坐标显示](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\gl_FragCoord坐标显示.png)

### `mix(x,y,a)`

return x * (1 - a) + y * a - liner interpolation;

x & y can be floats,vec2,vec3,vec4 etc.

只需要确保第一个数和第二个数类型相同，并且返回值匹配

e.g.

```
uniform vec2 u_resolution;
//    uniforms.u_resolution.value.x = window.innerWidth;
//    uniforms.u_resolution.value.y = window.innerHeight;

void main() {
  vec2 uv = gl_FragCoord.xy/u_resolution;
  vec3 color = mix(vec3(1.0,0.0,0.0),vec3(0.0,0.0,1.0),uv.x);
  gl_FragColor = vec4(color, 1.0);
}
```

![shaderMix展示图](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\shaderMix展示图.png)

### `mod(number, divisor)`

求余

can be floats,vec2,vec3,vec4 etc.

return number % divisor

### `clamp(n,minimum,maximum)`

takes 3 parameters

**`clamp()`** 函数的作用是把一个值限制在一个上限和下限之间，当这个值超过最小值和最大值的范围时，在最小值和最大值之间选择一个值使用

e.g:

- clamp(2.0,0.0,1.0) = 1.0

- clamp(-1.0,0.0,1.0) = 0.0

- ```js
  const fragmentShader = `
  varying vec3 v_position;
  uniform vec2 u_resolution;
  
  void main() {
    vec3 color = vec3(0.0);
    color.r = clamp(v_position.x,0.0,1.0);//如果没有clamp，那么会出现负值，但gl_FragColor处理时负值和0一样
    color.g = clamp(v_position.y,0.0,1.0);
    gl_FragColor = vec4(color, 1.0);
  }
  ```

  

![position显示样例](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\position显示样例.png)

### `step(edge , n)`

return edge < n ? 1.0 : 0.0

对比clamp，是不是异常显眼

e.g:

```js
const fragmentShader = `
varying vec3 v_position;
uniform vec2 u_resolution;

void main() {
  vec3 color = vec3(0.0);
  color.r = step(0.0,v_position.x);//如果没有clamp，那么会出现负值，但gl_FragColor处理时负值和0一样
  color.g = step(0.0,v_position.y);
  gl_FragColor = vec4(color, 1.0);
}
`
```

![image-20240419164047176](C:\Users\LEGION\AppData\Roaming\Typora\typora-user-images\image-20240419164047176.png)

### `smoothstep(edge0,edge1,n)`

```js
if(n < edge0){
	return 0.0
}else if(n > edge1){
	return 1.0
}else{
	return (x - edge0) / (edge1 - edge0)
}
```

e.g.

```js
const fragmentShader = `
varying vec3 v_position;
uniform vec2 u_resolution;

void main() {
  vec3 color = vec3(0.0);
  color.r = smoothstep(0.0,0.1,v_position.x);//如果没有clamp，那么会出现负值，但gl_FragColor处理时负值和0一样
  color.g = smoothstep(0.0,0.1,v_position.y);
  gl_FragColor = vec4(color, 1.0);
}
`

```

![smoothstep样例](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\smoothstep样例.png)

### `length(vec234)`

求长，若为vec2,3,4,求向量的模长

e.g.

```glsl
varying vec3 v_position;
uniform vec2 u_resolution;

void main() {
  vec3 color = vec3(0.0);
  float inCircle = 1.0 - step(0.5,length(v_position.xy));
  color = vec3(1.0,1.0,0.0) * inCircle;
  gl_FragColor = vec4(color, 1.0);
}
```

此时将点到(0.0,0.0,0.0)点距离和0.5相比，若大于0.5，则返回1(step())

v_position没设置的情况下，中心点为(0.0,0.0,0.0)

![circleShow](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\circleShow.png)

### `fract(genType x)`

以float x为例

```js
float fract(float x){
	if(x > 0){
		return mod(x,1);
	}else{
		return mod(abs(1 + x) , 1);
	}
}
//如 1.1 return 0.1， -1.1 return 0.9
```

### `atan(genType y,genType x )`

or genType atan( genType y_over_x);

返回反正切值，如`arctan(1) =  x => x = Π / 4；`

x ∈ [- Π / 2 , Π / 2  ]

### `floor(genType x)`

```c++
if(x > 0){
	return int(x);
}else{
	if(float(int(x)) == int(x) ){
		return int(x);
	}else{
		return int(x) - 1;
	}
}
```



## 原理

#### 顶点着色器

顶点着色器的目的是根据模型的移动方式将其移动到剪裁空间坐标中

顶点着色器的主函数main，必须设置gl_Position以及使用projectionMatrix(三维转二维)从而执行视图矩阵和顶点位置

##### 移动顶点过程(Transforming the vertex)：

position-local coordinate -> Move to world space -> Move to camera view -> Project on to screen

#### 片段着色器

片段着色器的主函数main，必须将值gl_FragColor设置为rgba的格式

片段着色器，调用屏幕空间中包含的每个像素上色(r,g,b,a) 



# texture材质渲染

最基础的材质展示

```js
const fragmentShader = `

varying vec3 v_position;
varying vec2 v_uv;

uniform sampler2D u_tex;

void main (void)
{
  vec3 color = texture2D(u_tex,v_uv).rgb;
  gl_FragColor = vec4(color, 1.0); 
}
`;
const assetPath = "https://s3-us-west-2.amazonaws.com/s.cdpn.io/2666677/";

const uniforms = {
  u_tex: {
    value: new THREE.TextureLoader().setPath(assetPath).load(
      'sa1.jpg'
    ),
  },
  u_time: { value: 0.0 },
  u_mouse: { value: { x: 0.0, y: 0.0 } },
};
const material = new THREE.ShaderMaterial( {
  uniforms: uniforms,
  vertexShader: vertexShader,
  fragmentShader: fragmentShader
} );
```

![texture示例](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\texture示例.png)

想使用灰度

` vec3 color = texture2D(u_tex,v_uv).rrr;`

![texture灰度示例](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\texture灰度示例.png)



```glsl
#define PI 3.141592653589
#define PI2 6.28318530718

varying vec3 v_position;
varying vec2 v_uv;

uniform sampler2D u_tex;

vec2 rotate(vec2 pt, float theta, float aspect){
  float c = cos(theta);
  float s = sin(theta);
  mat2 mat = mat2(c,s,-s,c);
  pt.y /= aspect;
  pt = mat * pt;
  pt.y *= aspect;
  return pt;
}

float inRect(vec2 pt, vec2 bottomLeft, vec2 topRight){
  vec2 s = step(bottomLeft, pt)- step(topRight, pt);
  return s.x * s.y;
}

void main (void)
{
  vec2 center = vec2(0.5);
  vec2 uv = rotate(v_uv - center , PI/2.0,2.0/1.5)+center;
  vec3 color = texture2D(u_tex,uv).rgb;
  gl_FragColor = vec4(color, 1.0); 
}
```





![texture等比例渲染示例](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\texture等比例渲染示例.png)



```glsl
vec2 rotate(vec2 pt, float theta, float aspect){
  float c = cos(theta);
  float s = sin(theta);
  mat2 mat = mat2(c,s,-s,c);
  pt.y /= aspect;
  pt = mat * pt;
  pt.y *= aspect;
  return pt;
}

void main (void)
{
  vec2 p = v_position.xy;
  float len = length(p);
  vec2 ripple = v_uv + p/len *0.03*cos(len*100.0-u_time*4.0);// 计算出一个涟漪效果的坐标偏移,len*100波纹数量，u_time*4.0速度
  float delta = (sin(mod(u_time,u_duration) * (2.0 * PI/u_duration))+1.0)/2.0;
  vec2 uv = mix(ripple,v_uv,delta);//delta变为0将变为连续的效果
  vec3 color = texture2D(u_tex , uv).rgb;
  gl_FragColor = vec4(color, 1.0); 
}
```



![texture涟漪示例](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\texture涟漪示例.png)







# 案例

## 使用自定义uniforms对fragmentShader进行控制

```js

//初步实现函数
const fragmentShader = `
uniform int u_time;//对于这里而言，想定义为double，float等都可
uniform vec3 u_color;
uniform vec2 u_resolution;
uniform vec2 u_mouse;


void main() {
  // float time = mod(u_time , 1000.0);
  vec3 color = vec3(u_mouse.x / u_resolution.x, (sin(float(u_time)) + 1.0)/2.0, u_mouse.y / u_resolution.y);
  // vec3 color = vec3((sin(u_time) + 1.0)/2.0,0.0,(cos(u_time)+1.0)/2.0);
  gl_FragColor = vec4(color, 1.0);
}
`
const uniforms = {
  u_time: { value: 0.0 },
  u_mouse: { value: new THREE.Vector2(0.0, 0.0) },
  u_resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
  u_color: { value: new THREE.Color(0xff0000) }
}

//使用这里进行关联
const material = new THREE.ShaderMaterial({
  uniforms:uniforms,
  vertexShader:vertexShader,
  fragmentShader:fragmentShader,
});

//使用监听器开启方法对uniforms进行动态更新
function move(event){
  uniforms.u_mouse.value.x = (event.touches) ? 
  event.touches[0].clientX : event.clientX;
  uniforms.u_mouse.value.y = (event.touches) ? 
  event.touches[0].clientY : event.clientY;
}

if('ontouchstart' in window){
  document.addEventListener('touchmove',move);

}else{
  window.addEventListener('resize',onWindowResize,false);
  document.addEventListener('mousemove',move);
}
//对window窗口尺寸大小进行热更新
function onWindowResize( event ) {
  const aspectRatio = window.innerWidth/window.innerHeight;
  let width, height;
  if (aspectRatio>=1){
    width = 1;
    height = (window.innerHeight/window.innerWidth) * width;
  }else{
    width = aspectRatio;
    height = 1;
  }
  camera.left = -width;
  camera.right = width;
  camera.top = height;
  camera.bottom = -height;
  camera.updateProjectionMatrix();
  renderer.setSize( window.innerWidth, window.innerHeight );
  if(uniforms.u_resolution != undefined){
    uniforms.u_resolution.value.x = window.innerWidth;
    uniforms.u_resolution.value.y = window.innerHeight;
  console.log(uniforms)
  }
}
//渲染时对u_time进行热更新
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame( animate );
  renderer.render( scene, camera );
  uniforms.u_time.value = clock.getElapsedTime();
}

```

### 使用uniforms实现颜色数值展示

```js
const uniforms = {
  u_color_a: { value: new THREE.Color(0xff0000) },
  u_color_b: { value: new THREE.Color(0x0000ff) },
  u_time: { value: 0.0 },
  u_mouse: { value:{ x:0.0, y:0.0 }},
  u_resolution: { value:{ x:0, y:0 }}
}

const fshader = `
uniform vec3 u_color_a;
uniform vec3 u_color_b;
uniform vec2 u_mouse;
uniform vec2 u_resolution;
uniform float u_time;

void main (void)
{
  float delta = (sin(u_time)+1.0)/2.0;
  vec3 color = mix(u_color_a, u_color_b, delta);
  gl_FragColor = vec4(color, 1.0); 
}
`

//因此在进行rgb显示的时候可以直接调用u_color_a等其他
```

## 片段着色器中调用顶点着色器显示的顶点 

注意，颜色通道采用介于0，1之间的值

### v_uv(screen position of a pixel)

geometry的attribute属性里面带有uv坐标

- 传递这点的U , V 值时从图象中选择位置的一种方式，在调用时，我们需要把顶点着色器中的U，V值保存为变化回u，v(小写)
- 在调用顶点着色器时，左下角为(0,0)右下角为(1,0),和`gl_fragCoord`同理![v_uv](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\v_uv.png)

- 如果发现每个像素都调用了片段着色器，那么我们就会在main方法之外声明它
- 那么，因为片段都是一个个三角形，那么应该那那个坐标来进行计算？
  - 我们需要对三个坐标进行插值运算(因此被称为`variing`，，，v in fragment shader),同时，片段着色器将接收透视(v_uv will be a blend of the uv values of each vertex in the triangle)

```js
const vertexShader = `
varying vec2 v_uv;

void main(){
  v_uv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position * 0.5, 1);
  
}
`
const fragmentShader = `
varying vec2 v_uv;
uniform vec2 u_resolution;

void main() {
  vec2 uv = gl_FragCoord.xy/u_resolution;
  vec3 color = vec3(v_uv.x,v_uv.y,0.0);
  gl_FragColor = vec4(color, 1.0);
}
`
```

![v_uv显示样例](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\v_uv显示样例.png)

### v_position （model position of a pixel）
![v_position坐标](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\v_position坐标.png)

```js
const vertexShader = `
varying vec3 v_position;

void main(){
  v_position = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position * 0.5, 1);
//这里的position * 0.5 意味着渲染坐标放缩0.5，但实际上还是原来的尺寸
  
}
`
const fragmentShader = `
varying vec3 v_position;
uniform vec2 u_resolution;

void main() {
  vec3 color = vec3(0.0);
  color.r = clamp(v_position.x,0.0,1.0);//如果没有clamp，那么会出现负值，但gl_FragColor处理时负值和0一样
  color.g = clamp(v_position.y,0.0,1.0);
  gl_FragColor = vec4(color, 1.0);
}
`
```

![position显示样例](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\position显示样例.png)

## 长方体自定义中心点边长

```js
const vertexShader = `
varying vec3 v_position;

void main(){
  v_position = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position * 0.5, 1);
  
}
`
const fragmentShader = `
varying vec3 v_position;
uniform vec2 u_resolution;

// pt给当前点的位置,size给当前向量(求长得),center给中心点
float inSquare(vec2 pt,vec2 size,vec2 center){
  if(length(size.x) >= abs(center.x - pt.x) && length(size.y) >= abs(center.y - pt.y)){
    return 1.0;
  } 
  return 0.0;
}

void main() {
  vec3 cenCoordinate = vec3(0.25,0.5,0.0);//定义中心点
  vec2 WandH = vec2(0.25,0.5); //定义长宽(一半)
  float inRect = inSquare(v_position.xy,WandH.xy,cenCoordinate.xy);
  vec3 color = vec3(0.0,1.0,1.0) * inRect;
  if(length(color) == 0.0 ) 
    gl_FragColor = vec4(0.0,1.0,0.0,1.0);
  else 
    gl_FragColor = vec4(color,1.0);

  if(length(v_position) < 0.01 || length(v_position - cenCoordinate) < 0.01) gl_FragColor = vec4(1.0,1.0,1.0,1.0);//给出需要高亮的点
}
`
```

![rectangle样例](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\rectangle样例.png)

### 基础上添加绕中心轨道旋转和自选转

```js
const vertexShader = `
varying vec3 v_position;

void main(){
  v_position = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position * 0.5, 1);
  
}
`
const fragmentShader = `
varying vec3 v_position;
uniform vec2 u_resolution;
uniform float u_time;

mat2 getRotationMatrix(float theta){
  float s = sin(theta);
  float c = cos(theta);
  return mat2(c,-s,s,c);
}

// pt给当前点的位置,size给当前向量(求长得),center给中心点
float inSquare(vec2 pt,vec2 size,vec2 center){
  if(length(size.x) >= abs(center.x - pt.x) && length(size.y) >= abs(center.y - pt.y)){
    return 1.0;
  } return 0.0;
}

void main() {
  vec3 cenCoordinate = vec3(0.25 * sin(u_time),0.5 * cos(u_time),0.0);//定义中心点
  mat2 mat = getRotationMatrix(u_time);//获取旋转矩阵
  vec2 WandH = vec2(0.25,0.5); //定义长宽(一半)
  float inRect = inSquare(mat * v_position.xy ,WandH.xy,mat * cenCoordinate.xy);
  vec3 color = vec3(0.0,1.0,1.0) * inRect;
  if(length(color) == 0.0 ) 
    gl_FragColor = vec4(0.0,1.0,0.0,1.0);
  else 
    gl_FragColor = vec4(color,1.0);

  if(length(v_position) < 0.01 || length(v_position - cenCoordinate) < 0.01) gl_FragColor = vec4(1.0,1.0,1.0,1.0);//给出需要高亮的点
}
`
```

![自选转样例](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\自选转样例.png)l

### 基础上添加放缩

```js
const fragmentShader = `
varying vec3 v_position;
uniform vec2 u_resolution;
uniform float u_time;

mat2 getRotationMatrix(float theta){
  float s = sin(theta);
  float c = cos(theta);
  return mat2(c,-s,s,c);
}

// pt给当前点的位置,size给当前向量(求长得),center给中心点, bias偏移值(自旋转的中心点的向量->用于计算需要旋转的中心点)
float inSquare(vec2 pt,vec2 size,vec2 center,vec2 bias){
  if(length(size.x) >= abs(center.x + bias.x - pt.x) && length(size.y) >= abs(center.y + bias.y - pt.y)){
    return 1.0;
  } return 0.0;
}

void main() {
  vec3 cenCoordinate = vec3(0.5,0.0,0.0);//定义中心点
  mat2 mat = getRotationMatrix(u_time);//获取旋转矩阵
  vec2 WandH = vec2(0.125,0.25); //定义长宽(一半)
  vec2 bias = vec2(0.125,0.25);//偏置值，自旋转的中心点的向量->用于计算需要旋转的向量
  // mat2 scale = mat2((sin(u_time) + 1.0) / 8.0 , 0 ,(cos(u_time) + 1.0) / 8.0 , 0);
  float scale =clamp( (sin(u_time) + 2.0) / 2.0,0.0,1.5);


  float inRect = inSquare( mat * v_position.xy , scale * WandH.xy, mat * cenCoordinate.xy, scale * bias);
  vec3 color = vec3(0.0,1.0,1.0) * inRect;
  if(length(color) == 0.0 ) 
    gl_FragColor = vec4(0.0,1.0,0.0,1.0);
  else 
    gl_FragColor = vec4(color,1.0);

  if(length(v_position) < 0.01 || length(v_position - cenCoordinate) < 0.01) gl_FragColor = vec4(1.0,1.0,1.0,1.0);//给出需要高亮的点
}
`
```

![放缩](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\放缩.png)

### 镂空圆和边缘雾化

```js
const fragmentShader = `
#define PI2 6.28318530718

uniform vec2 u_mouse;
uniform vec2 u_resolution;
uniform float u_time;

varying vec2 v_uv;
varying vec3 v_position;

// pt:点的模型坐标, center:中心点, radius:半径, soften:开启边缘雾化
float circle(vec2 pt ,vec2 center, float radius, bool soften){
  vec2 p = pt - center;
  float edge = (soften) ? radius * 0.05 : 0.0; 
  return 1.0 - smoothstep(radius-edge,radius+edge , length(p));
}

// 中心镂空圆
float circle(vec2 pt ,vec2 center, float radius, float line_width){
  vec2 p = pt - center;
  float len = length(p);
  float half_line_width = line_width / 2.0;
  return step(radius - half_line_width,len) - step(radius + half_line_width,len);
}

void main (void)
{
  vec3 color = vec3(1.0,0.0,0.0) * circle(v_position.xy,vec2(0.5),0.3,0.02);
  gl_FragColor = vec4(color, 1.0); 
}
`
```

![镂空圆和圆边缘雾化](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\镂空圆和圆边缘雾化.png)

## 线

实现思路：

当B在正负一半线路之间时，那么就在线上

同时使用平滑补偿来处理交线

```js

const vertexShader = `
varying vec2 v_uv;
varying vec3 v_position;
void main() {	
  v_uv = uv;
  v_position = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 2 );
}
`
const fragmentShader = `
#define PI2 6.28318530718

uniform vec2 u_mouse;
uniform vec2 u_resolution;
uniform float u_time;

varying vec2 v_uv;
varying vec3 v_position;


float rect(vec2 pt,  vec2 size, vec2 center){
  vec2 p = pt - center;  
  vec2 halfsize = size * 0.5;
  float horz = step(-halfsize.x , p.x) - step(halfsize.x , p.x);
  float vert = step(-halfsize.y , p.y) - step(halfsize.y , p.y);
  return horz * vert;
}


float line(float a, float b, float line_width,float edge_thickness){
  float half_line_width = line_width * 0.5;
  return smoothstep(a - half_line_width - edge_thickness, a - half_line_width , b ) - 
  smoothstep(a + half_line_width, a + half_line_width + edge_thickness, b);
  //因为中心点是(0.0,0.0,0.0),因此需要+，-
}

void main (void)
{
  vec2 uv = v_uv;
  vec3 color = vec3(1.0) * line(v_position.y,mix(-0.8,0.8,(sin(v_position.x * 
  3.1415) + 1.0) / 2.0),0.05,0.002);
  gl_FragColor = vec4(color, 1.0); 
}
`

```





![线展示图](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\线展示图.png)

#### 绕边的点自旋转的线

```glsl
/**
 * @description: 
 * @param {*} pt v_uv，屏幕位置
 * @param {*} center 旋转中心点
 * @param {*} radius 控制半径长度
 * @param {*} line_width 线宽
 * @param {*} edge_thickness 边缘厚度
 * @return {*}
 */
float sweep(vec2 pt,vec2 center, float radius, float line_width, float edge_thickness){
  vec2 d = pt - center;//当前点到圆心的向量
  float theta = u_time * 2.0;
  vec2 p = vec2(cos(theta), -sin(theta)) * radius;//当前应当呈现的向量
  float h = clamp (dot(d,p) / dot(p,p),0.0,1.0);//求h在p上的投影长度比
  float l = length(d - p*h);//p * h是d投影到p上的线，d - p * h 就是求出当前点到p的那条线上的垂直的线
  return 1.0 - smoothstep(line_width , line_width + edge_thickness,l);
}

```

![自旋转的线](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\自旋转的线.png)

## 自定义边长求多边形

```c++
/**
 * @description: 360°对边数量进行切分,得出角度,加上边长形成边,得出结果
 * @param {*} pt 当前点的位置
 * @param {*} center 圆心
 * @param {*} radius 半径
 * @param {*} sides 边的数量
 * @param {*} rotate 旋转角度
 * @param {*} edge_thickness 边缘厚度
 * @return {*} 显示比
 */
float polygon(vec2 pt, vec2 center, float radius, int sides, float rotate, float edge_thickness){
  pt -= center;// 得当前点离中心点向量
  float theta = atan(pt.y, pt.x) + rotate;//得出当前角度
  float rad = PI2/float(sides);//得出每条边占据的角度
  float d = cos(floor(0.5 + theta/rad)*rad  -theta)*length(pt);//计算了当前点到多边形边缘的距离 
// theta/rad = 多边形的第几条边， floor(0.5 + ...) => 四舍五入 *rad => 当前边所在的 cos=
  return 1.0 - smoothstep(radius, radius + edge_thickness, d);
}
```

![多边形形成样例](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\多边形形成样例.png)



## 伪噪声

```glsl
float random(vec2 st, float seed){
  const float a = 1.0;
  const float b = 1.0;
  const float c = 10.0;
  return fract(sin(dot(st,vec2(a,b)) + seed) * c);
}

void main (void)
{
  vec3 color = random(v_uv, u_time / 10) * vec3(1.0);
  gl_FragColor = vec4(color, 1.0); 

}
```

![斜面线模拟随机](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\斜面线模拟随机.png)

### 平滑点噪声

```glsl
uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

varying vec2 vUv;

// 2D Random
float random (vec2 st) {
    return fract(sin(dot(st, vec2(12.9898,78.233)))
                 * 43758.5453123);
}

float noise (vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    // Smooth Interpolation

    // Cubic Hermine Curve.  Same as SmoothStep()
    vec2 u = f*f*(3.0-2.0*f);
    // u = smoothstep(0.,1.,f);

    // Mix 4 coorners percentages
    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

void main() {
    vec2 st = vUv;
    vec2 pos = vec2(st*8.0);
    pos.y -= u_time;
    float n = noise(pos);

    gl_FragColor = vec4(vec3(n), 1.0);
}
```



![平滑点噪声展示](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\平滑点噪声展示.png)

### 像素火

```glsl
uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;
uniform vec3 u_color_a;
uniform vec3 u_color_b;

varying vec2 vUv;

// 2D Random
float random (vec2 st) {
    return fract(sin(dot(st, vec2(12.9898,78.233)))
                 * 43758.5453123);
}

// 2D Noise based on Morgan McGuire @morgan3d
// https://www.shadertoy.com/view/4dS3Wd
float noise (vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    // Smooth Interpolation

    // Cubic Hermine Curve.  Same as SmoothStep()
    vec2 u = f*f*(3.0-2.0*f);
    // u = smoothstep(0.,1.,f);

    // Mix 4 coorners percentages
    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

void main() {
    vec2 n = vec2(0.0);
    vec2 pos;

    //Generate noise x value
    pos = vec2(vUv.x*1.4 + 0.01 , vUv.y - u_time*0.69);
    n.x = noise(pos*12.0);
    pos = vec2(vUv.x*0.5 - 0.033, vUv.y*2.0 - u_time*0.12);
    n.x += noise(pos*8.0);
    pos = vec2(vUv.x*0.94 + 0.02, vUv.y*3.0 - u_time*0.61);
    n.x += noise(pos*4.0);

    // Generate noise y value
    pos = vec2(vUv.x*0.7 - 0.01, vUv.y - u_time*0.27);
    n.y = noise(pos*12.0);
    pos = vec2(vUv.x*0.45 + 0.033, vUv.y*1.9 - u_time*0.61);
    n.y += noise(pos*8.0);
    pos = vec2(vUv.x*0.8 - 0.02, vUv.y*2.5 - u_time*0.51);
    n.y += noise(pos*4.0);
    
    n /= 2.3;


    vec3 color = mix(u_color_a, u_color_b, n.y*n.x);

    gl_FragColor = vec4(color, 1.0);
}
```

![像素火实例](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\glsl入门\像素火实例.png)

## 使用噪波来创建木材和大理石噪波







 





