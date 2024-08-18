论文报刊

SIGGRAPH Asia and ToG

# 第一节

# 第二节

#### Basic Graphics(Hardware) Pipeline

![图形管线渲染流程](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\2.图形管线渲染流程.png)

![细化步骤](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\2.细化步骤.png)

### OpenGL

- Is a set of APIs that the GPU pipeline from GPU

  - **Therefore, language does not matter**
  - Cross platform
  - Alternatives

- Cons

  - Fragmented: lots of different versions
  - **<u>C</u>** styles, not easy to use

- HOW ( like oil painting )

  1. Place objects

     - Model specification
     - Model transformation
     - <u>**User specifies an object's vertices, normals, texture,coords and send them to GPU as a Vertex buffer object(VBO)**</u>
       - Very similar to . obj files
     - <u>**Use OpenGL functions to obtain matrices**</u>
       - e.g.(exempli gratia), glTranslate, glMultMatrix, etc.(等)
       - No need to write anything on your own

  2. Set position of an easel

     - View transformation
     - Create / use a **framebuffer**
     - Set camera (the viewing transformation matrix) by simply calling, e.g. , gluPerspective(fovy aspect zNear zFar)

  3. Attach a canvas to the easel

     - Analogy of oil painting

     - one rendering **pass** in OpenGL
       - A framebuffer is specified to use
       - Specify one or more textures as output(shading, depth, etc)
       - 一般不直接渲染到屏幕（上一帧没渲染完成下一帧就开始渲染了）
       - Render (fragment shader specifies the content on each texture)

  4. Paint to the canvas

     - i.e.,how to perform shading
     - This is when vertex /fragment shaders will be used
     - For each vertex in parallel
       - OpenGL calls user-specified vertex shader: Transform vertex
     - For each fragment in parallel
       - OpenGL calls user-specified fragment shader: Shading and lighting calculations
       - OpenGL handles z-buffer depth test unless overwritten
     - Summary: in each pass
       - Specify objects,camera,MVP,etc
       - Specify framebuffer and input/output textures
       - Specify vertex / fragment shaders
       - (When you have everything specified on the GPU) Render

  5. (Attach other canvases to the easel and continue painting)

  6. (Use previous paintings for reference)

##### OpenGL Shading Language(GLSL)

### The Rendering Equation

![Rendering Equation](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\2.Rendering Equation.png)



#### Environment Lighting

- Representing incident lighting from all directions
  - Usually represented as a cube map or a sphere map (texture)
  - We'll introduce a new representation in this course



### Calculus(微积分)

# 第三节

## Recap: shadow mapping

A 2-Pass Algorithm(双端光线追踪 : bi-directional ray tracing)

- The <u>light pass</u> generates the SM
- The <u>camera pass</u> uses the SM(recall last lecture)

An image-space algorithm 

- Pro: no knowledge of scene's geometry is required
- Con: causing self occlusion(自遮挡) and aliasing issues

Well known shadow rendering technique

- Basic shadowing technique even for early offline rendering e,g. toy story

The math behind shadow mapping 

-  Project visible points in eye view back to light source -> will have a hard shadow
  - have a shadow, can easily find the mesh relation in 3D space

![3.双端光线追踪中人眼中能看到的但没接收光直射及镜面反射的点](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\3.双端光线追踪中人眼中能看到的但没接收光直射及镜面反射的点.png)

视线求交，

- Projection the depth map onto the eye's view

![3.点光源深度图象展示](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\3.点光源深度图象展示.png)

怎么理解透视投影？

 当成两步，先把透视投影挤压成平行投影，然后在挤压成一个长方形，然后推平，用z值去比较，z-buffer，自近向远平面移动

或者用距离比较，物体像素+光线向量+点光源(or camera)位置->距离

self occlusion(自遮挡 : 自身挡住了身体的一部分)

 ![3.自遮挡+数字精度产生的摩尔纹](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\3.自遮挡+数字精度产生的'摩尔纹'.png)

距离越远时，一个像素所代表的区域越大，一个像素表示了一个区域的"平均像素",这也是相机下的远距离的可能产生的摩尔纹(采样问题)的原因

![3.光线阴影所描述的场景](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\3.光线阴影所描述的场景.png)

这种情况下，实际上人能见到的的一个像素点是红色的一条线(镜面反射)，光到地上的蓝色线，实际上是光线深度

shadow map 记录的深度不同，因为深度不连续(阴影拉长，平面像素点**平行**)，导致了场景会进行一个曲解 (detached shadow)：部分解决方案

Adding a (variable) <u>bias</u> to reduce self occlusion(工业界常用方法)

- But introducing **detached shadow** issue 

Second-depth shadow mapping

- Using the **midpoint** between first and second depths  in SM (多存一个最小深度的一小深度，典型空间换)
- Unfortunately , requires objects to be **watertight**.
- And the **overhead** may not worth it.

![浅深度，次浅深度](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\浅深度，次浅深度.png)

***RTR(实时渲染 Real Time Rendering) does not trust in COMPLEXITY***，it just believe the **absolute** speed 

Issues in Shadow Mapping 

- Aliasing

## Project to light for shadows 

Inequalities in Calculus e.g (RTR的奇技淫巧)

- Schwarz不等式
- Minkowski不等式
- 切比雪夫不等式

--->  

In RTR, we care more about "approximately equal"

An important approximation throughout RTR

![3.重要的近似公式在RTR中](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\3.重要的近似公式在RTR中.png)

tips : (其实左边就是求平均值)

- When is it (more) accurate?
  - 实际积分域特别小，the support(Ω) is so small
  - G(x) is smooth，G(x)变化不大,not (光滑)

![3.拆相似公式](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\3.拆相似公式.png)

右边积分拆开，直接工作量降一截

When is it accurate

- One visibility,one vector, Small support  e.g.
  - <u>point / directional lighting</u>
- Smooth integrand(G(x)) e.g 
  - <u>Diffuse bsdf / constant radiance area lighting</u>

- An~~d We'll see it again in Ambient Occlusions, etc

## Percentage closer soft shadows

From Sard Shadows to Soft Shadow

![3.软硬阴影](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\3.软硬阴影.png)

**Percentage Closer Filtering(PCF)**

百分比渐进过滤

- Provides anti-aliasing at <u>shadows'edges</u>
  - Not for soft shadows(PCSS is, introducing later)
  - Filtering the result of shadow comparisons
- Why not filtering the shadow map?
  - Texture filtering just averages color components(颜色分量), i.e. you'll get blurred shadow map first
  - Averaging depth values, then comparing you still get a <u>binary</u> visibility(用二进制表示的可见度)

- Solution

  ![3.PCF实现办法](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\3.PCF实现办法.png)

->

![3.0 PCF效果](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\3.0 PCF效果.png)

- -> filter 的大小就决定了选择软硬阴影
  - Small -> sharper -> Hard Shadow
  - Large -> softer -> Soft Shadow
- Can we use PCF to achieve soft shadow effects?
- Key thoughts
  - From hard shadows to shft shadows
  - What's the correct size to filter?
  - Is it uniform?
    - Key observation
    - Where is sharper? Where is softer?(远的地方多软阴，近的地方多硬阴影，和遮挡物的距离有关)
- Key conclusion
- ![3.软硬阴影硬解](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\3.软硬阴影硬解.png)

- Now the only question
  - What's the blocker depth d_Blocker(考虑平均的blocker深度)
  - The complete algorithm of PCSS
    - Step1: Blocker search
      - (getting the average blocker depth in a certain region)
    - Step 2: Penumbra estimation
      - (Use the average blocker depth to determine filter size)
    - Step 3: Percentage Closer Filtering
- Which region to perform blocker search?
  - can be set constant(常量)(e.g. 5*5),but can be better with heuristics

![3.ShadowPoint简易解决办法](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\3.ShadowPoint简易解决办法.png)

(以上，开销巨大)

shadow mapping只能处理单光源问题

## Basic filtering techniques

 



# 第四节

## More on PCF and PCSS

![4.PCF中的算法](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\4.PCF中的算法.png)

![4.PCF中的算法2](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\4.PCF中的算法2.png)

Revisiting PCSS

- THe complete algorithm of PCSS
  - Step 1 : Blocker search
    - (getting the average blocker depth in a certain region)获取区域中平均阻塞深度
  - Step 2 : Penumbra estimation\
    - (use the average blocker depth to determine filter size)使用第一步中的平均阻塞深度图来确定过滤器
  - Step 3 : Percentage Closer Filtering（使用过滤器进行筛选）
- Which step(s) can be slow?
  - Looking at every texel inside a region (steps 1 and 3)
  - Softer -> large filtering region -> slower

## Variance soft shadow mapping

实时软阴影

- Faster blocker search (step 1) and filtering (step 3)（针对性的解决）

- “percentage closer” filtering qs

  - the percentage of texels that are in front of the shading point
  - how many texels are closer than t in the search area,i.e
  - how many student did better than you in an exam

  ![4.分布图](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\4.分布图.png)

使用正太分布(计算均值(mean)和方差(variance))来得到一个近似的分布（or 高斯分布）

- 平均值和方差的解决方法(Can be handled by MipMap and Summed Area Table(SAT)求和面积表)

  - MipMaping 但是要求正方形，且同时不同层级之间要做插值来平衡
  - Summed Area Tables(SAT)
  - Variance
    - Var(X) = E(X²) - E²(x)
      - 概率论牛逼！
    - 因此我们只需要额外的一个depth²的通道
  - ![4.正态分布计算](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\4.正态分布计算.png)
  - 兄弟直接概率论嘎嘎得结论
  - 积分出来得值打成的表叫 Error function 误差函数

  切比雪夫不等式(oh 我那美丽的世界线)

  ![4.切比雪夫不等式](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\4.切比雪夫不等式.png)

在实时渲染中，不考虑不等，考虑约等

总结：

Performance

- shadow map generation
  - "square depth map": parallel, along with shadow map
- Run time
  - Mean of depth in a range: O(1)
  - Mean of depth square in a range: O(1)
  - Chebychev(分布，均值方差推): O(1)
  - 不需要循环，遍历一遍够了
- Step 3 (filtering) solved perfectly
  - 每一帧新生成的情况下，都要做一遍shadow maping，因此还是有一定开销， MipMap 通常生成非常快，开销不用考虑



返回步骤一：在遮挡物搜索中

- Also require sampling(loop) earlier, also inefficient
- The average depth of blockers
- Not the average depth Zavg(Z轴上的平均深度)

Key idea

- Blocker (z < t),avg, Zocc (Z-Occlude)
- Non-blocker(z>t), avg, Zunocc(Z在shadow mapping记录的深度，要大于我的shadding point的渲染深度)

![4.遮挡物数量图](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\4.遮挡物数量图.png)

![4.遮挡物平均](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\4.遮挡物平均.png)

- Approximation: N1/N = P(x  >  t),Chebychev
- Approximation: N2/N = 1 - P(x > t)
- 对于Zunocc而言，并不能真实知道
  - 因此，在不知道的情况下，我们通常把Zunocc 默认为shading point的深度（？->）
  - 对于目标而言，是采样点中心，它的采样点范围是个平面，因此直接默认其实问题不大(雾)
- Step 1 solved with negligible additional cost(以可忽略不计的额外成本解决问题)

## MIPMAP and Summed-area Variance shadow Maps

最简单的做法，还是MIPMAP

- 快速，近似，方形查询
- still approximate even with trilinear interpolation(三线性插值)
  - 当使用层与层之间的查询时，就需要使用线性插值。因此只能近似。各向异性过滤能解决。

在一维数组上处理 Summered area Variance-> 

![4.SAT的范围查询](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\4.SAT的范围查询.png)

- 如想前数中间三个，那么只需要第六个20 - 第三个9就可以了

- 简而言之，预计算，前缀和思路

在二维数组上处理Summered area Variance

![4.SAT的二维范围查询](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\4.SAT的二维范围查询.png)

- 这这张表的任意一个元素，表示从左到右从上到下加起来的值的总和，比如(3,3),那么表达的值就是(0,0)到(3,3)的长方体范围内产生的和。如何生成：

  - 简单思路

    ```c++
    int L,W;//长宽
    int a[L,W],SAT[L,W];// 简单写，假设a以读值,SAT是最终要的求和面积表
    for(int i = 0;i<W;i++){//自上而下
        for(int j = 0;j<L;j++){
    		if(i = 0){
                SAT[j , i] = a[j , i];
            }else{
                SAT[j , i] = a[j , i] + a[j , i - 1];
            }
        }
    }
        
    ```

    

- Note: accurate, but need O(n) time and storage to build

  - Storage might not be an issue

## Moment shadow mapping

 ![4.VSSM缺点](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\4.VSSM缺点.png)



当渲染的时候，在上面这个视角射线计算过去的分布，会在交界处会有较大的覆盖物(暂时这么叫)，从而有多个分布值，从而不能用高斯分布来计算

- Issues if the edpth distribution is inaccurate
  - Overly dark: may be aceeptable
  - Oright bright: **light Leaking**
  - ![4.vssm挡不住的例子](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\4.vssm挡不住的例子.png)

原本暗的地方更暗没感觉(能，，，忍)，在一块地方突然亮直接非常明显(漏光)

![4.vssm漏光](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\4.vssm漏光.png)

- Goal
  - Represent a distribution more accurately
    - (but still not too costly to store)
- Idea
  - Use **higher order moments** to represent a distribution

- Moments
  - Quite a few variations on the definition
  - We use the simplest:
    - x,x²,x³,x⁴...（like泰勒，越多越接近期望值）
  - So,VSSM is essentially using the **first two** orders of moments
  - What can moments do?
    - ![4.CDF回复](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\4.CDF回复.png)

通常四阶

工业界做法：如果不考虑精度的话，常见为packing和unpacking，(定点量化技术？)如果这么做，就不能插值了，两个16位拼成一块后，虽然能用，但是位置信息全乱了。



# 第五节

## Finishing up on shadows

### <u>Distance field soft shadows</u>

SDF(Signed Distance Field) 难贴图

- 快，但需要大量的存储

![5.distanceFielf效果比较](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\5.distanceField效果比较.png)

GAMES101 距离算法

![5.距离算法展示](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\5.距离算法展示.png)

The Usages of Distance fields

- 使用方法一：球形光线追踪(Sphere Distance Fields)
  - 将空的空间场分为使用球型进行分配，当光线接触球体时直接跳过球体
- 使用方法二：使用角度来说明是否有障碍物("sage angle")
  - 这个角度越小，也就意味着能看到的空白空间越小
  - ![5.光线追踪安全角度使用方法示意](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\5.光线追踪安全角度使用方法示意.png)
  - 如何写并且如何实现软硬阴影控制
  - ![5.实用角度光线追踪](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\5.实用角度光线追踪.png)



## Shading from environment lighting

- Informally named <u>Image-Based Lighting(IBL)</u>
- How to use it to shade a point (without shadows)
  - solving the rendering equation

- General solution - Monte Carlo integration
  - Numerical
  - Large amount of samples required
- Observation
  - if the BRDF(Bidirectional Reflectance Distribution Function) is glossy - small support
  - if the BRDF is diffuse - smooth
  - ![5.BRDF公式](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\5.BRDF公式.png)

- Prefiltering of the environment lighting
- - Pre-generation a set of differently filtered environment lighting
  - Filter size in-betewwn can be approximated via trilinear interp
  - 可以先提前计算filter，然后计算时，取light和镜面反射方向就行，然后使用插值的方式完善
  - BRDF Lobe 的形状越尖锐，即环境光积分范围越小，就需要使用模糊程度更低的环境贴图；反之 BRDF Lobe 的形状越粗壮，即环境光积分范围越大，就需要使用模糊程度更高的环境贴图
  - 可用 <u>Mipmap</u> 来生成不同Level的环境贴图，通过三线性插值的方式来得出任何模糊程度且任何2D位置的环境光滤波结果
- The second term is still an integral
  - How to avoid sampling this term
  - ![5.进一步计算](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\5.进一步计算.png)

## shadow from environment lighting



# 第六节

实时环境映射

## Precomputed Radiance Transfer(PRT)预计算辐射性光线传递

# 作业

## blinn-phong 实现

![h1.blinn-phong公式](C:\Users\LEGION\Desktop\笔记\计算机图形学\图包\GAMES202-高质量实时渲染\h1.blinn-phong公式.png)

```c++
#ifdef GL_ES
precision mediump float;//指定着色器的精度，使用中等精度的float
#endif
uniform sampler2D uSampler;//定义了2D纹理采样器
//binn
uniform vec3 uKd;//定义漫反射系数
uniform vec3 uKs;//定义镜面反射系数
uniform vec3 uLightPos;//定义光源位置
uniform vec3 uCameraPos;//定义相机位置
uniform float uLightIntensity;//定义光照强度
uniform int uTextureSample;//定义纹理采样标志

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

void main(void){
  vec3 color;
  if(uTextureSample == 1){
    color = pow(texture2D(uSampler, vTextureCoord).rgb,vec3(2.2));
  }else{
    color = uKd;
  }//uTextureSample(纹理标志)的值来决定使用纹理采样的颜色还是漫反射系数的颜色。如果uTextureSample为1，就对纹理采样的颜色进行gamma矫正后赋给color；否则使用漫反射系数的颜色。

  vec3 ambient = 0.05 * color;//环境光分量，它是颜色的5%。

// 下面是漫反射处理
  vec3 lightDir = normalize(uLightPos - vFragPos);//光线方向
  vec3 normal = normalize(vNormal);//单位化
  float diff = max(dot(lightDir, normal),0.0);//定义漫反射系数，dot内积
  float light_atten_coff = uLightIntensity / length(uLightPos - vFragPos);//光照衰减系数
  vec3 diffuse = diff * light_atten_coff * color;//最终的漫反射光颜色

//下面是镜面反射处理
  vec3 viewDir = normalize(uCameraPos - vFragPos);//计算镜面反射光分量
  float spec = 0.0;
  vec3 reflectDir = reflect(-lightDir, normal);//计算反射方向
  spec = pow(max(dot(viewDir,reflectDir),0.0),35.0);//计算镜面反射强度
  vec3 specular = uKs * light_atten_coff * spec;//得到镜面反射光颜色


  gl_FragColor = vec4(pow((ambient + diffuse + specular),vec3(1.0/2.2)),1.0);//对环境光/漫反射光和镜面反射光叠加后进行gamma矫正，把结果赋给内置的gl_FragColor变量，并且最终表示这个像素的最终颜色输出

}
```

