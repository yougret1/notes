# Node.js

配套代码：node-learn



## 简要

### 1.Node.js可以做什么

- 基于Express框架，可以快死构建Web应用
- 基于Electron框架，可以构建跨平台的桌面应用
- 基于restify框架，可以快速构建API接口应用
- 读写和操作数据库，创建实用的命令行辅助前端开发，etc

### 命令

#### 查看版本号

`node -v`

#### 用node执行文件

`node 文件名`

## fs文件系统模块,I/O 

`fs.readFile()`方法，用于读取指定文件中的内容

`fs.writeFile()`方法，用来向指定的文件中写入内容

- writeFile,文件未找到时，会自动创建一个文件，后缀名相同
- 在读取的时候 `__dirname`表示当前文件存放的目录
- Tip，只能创建文件，不能创建路径(文件夹)

```js
// 导入fs模块
const fs = require('fs')  // 相当于 <script src='./fs></script>
// 2、fs.readFile('文件的路径','编码格式',回调函数)
// fs.readFile('./test1.js','utf8',(err,data)=>{
fs.readFile('./test.js', 'utf8', (err, data) => {
    // 读取成功 err为null
    // 读取失败 err为错误对象
    if (err) return console.log(err.message);
    console.log("read success!" + data);
})
```

##### 路径拼接 path.join

The `path.join()` method joins all given `path` segments together using the platform-specific separator as a delimiter, then normalizes the resulting path.

Zero-length `path` segments are ignored. If the joined path string is a zero-length string then `'.'` will be returned, representing the current working directory.

```js
path.join('/foo', 'bar', 'baz/asdf', 'quux', '..');
// Returns: '/foo/bar/baz/asdf'

path.join('foo', {}, 'bar');
// Throws 'TypeError: Path must be a string. Received {}' 
```

##### 获取路径最后一个部分 path.basename()

该`path.basename()`方法返回 a 的最后一部分`path`

`path.extname()`方法返回最后一个的扩展名

```js
path.basename('/foo/bar/baz/asdf/quux.html');
// Returns: 'quux.html'

path.basename('/foo/bar/baz/asdf/quux.html', '.html');//第二个参数时文件扩展名

// Returns: 'quux'//如果后缀名不匹配，那么依旧返回quux.html
```

## http模块

http模块是Node.js官方提供的，用来创建web服务器的模块，通过http模块提供的`http.creatServer()`方法，就可以把一台电脑变为一台Web服务器，从而对外提供Web资源服务

- 服务器和普通电脑的区别在于，服务器上安装了web服务器软件，例如IIS、Apache等。通过安装这些服务器软件，就能把一台普通的电脑变成一台web服务器

#### 基本的未能返回任何值得服务器代码

```js
// 导入http模块
const http = require("http");
//创建web服务器实例
const server = http.createServer();
//为服务器实例绑定request时间，监听客户端的请求
server.on("request", function (req, res) {
  // req 是请求对象，包含了客户端相关的数据和属性
  // req.url是客户端请求的URL地址
  // req.method 是客户端的method请求类型
  console.log("Someone visit our web server");
  return null
});
//启动服务器
server.listen(8085, function (param) {
  console.log("server running at 8085");
});
```

#### 解决中文乱码问题

当调用 `res.end()`方法，向客户端发送中文内容的时候，会出现乱码问题，此时需要手动设置内容

## module模块

每个.js自定义模块中都有一个module对象，它里面存储乐和当前模块有关的信息，打印如下

```js
Module {
  id: '.',
  path: 'd:\\youguangtao gitee文件备份\\node-learn\\04.模块作用域',
  exports: {},
  filename: 'd:\\youguangtao gitee文件备份\\node-learn\\04.模块作用域\\test.js',
  loaded: false,
  children: [
    Module {
      id: 'd:\\youguangtao gitee文件备份\\node-learn\\04.模块作用域\\模块作用域.js',
      path: 'd:\\youguangtao gitee文件备份\\node-learn\\04.模块作用域',
      exports: {},
      filename: 'd:\\youguangtao gitee文件备份\\node-learn\\04.模块作用域\\模块作用域.js',
      loaded: true,
      children: [],
      paths: [Array]
    }
  ],
  paths: [
    'd:\\youguangtao gitee文件备份\\node-learn\\04.模块作用域\\node_modules',
    'd:\\youguangtao gitee文件备份\\node-learn\\node_modules',
    'd:\\youguangtao gitee文件备份\\node_modules',
    'd:\\node_modules'
  ]
}
```

### `module.exports` 对象

在歪解使用require导入一个自定义模块的时候，得到的对象就是那个模块中，通过module.export指向的那个对象

### exports对象  （指向module.exports）

由于`module.exports`单词写起来比较复杂，为了简化向外共享成员的代码，Node提供了`exports`对象，默认情况下`exports`和`module.exports`指向同一个对象。最终共享结果还是以`module.exports`指向的对象为准

## npm与包

https://www.npmjs.com/中搜索自己所需要的包

https://registry.npmjs.org/服务器上下载自己所需要的包

### npm初体验

`node_modules` 文件夹用来存放所有已安装到项目中的包。require()导入第三方包的时候，就是从这个目录中查找并加载包

`package-lock.json`配置文件用来记录node_modules目录下的每一个包的下载信息，例如包的管理，版本号，下载地址等。

### 安装指定版本的包

```js
npm i moment@2.22.2
```

- 第一位数字，大版本
- 第二个数字，功能版本
- 第三个数字，Bug修复版本

版本号提升规则，只要前面的版本号增长了，则后面的版本号归零

### package.json——包管理配置文件

用来记录项目有关的一些配置文件，例如

- 项目的名称，版本号，描述
- 项目中都用到了那些包
- 那些包旨在开发期间会用到
- 那些包在开发和部署的时候都需要用到

### devDependencies节点

如果某些包旨在项目开发阶段用到，在项目上线后不会用到，那么则建议将这些包记录到devDependencies节点中，与之对应的，如果某些包在开发和项目和项目上线之后都需要用到，则建议把这些包记录到dependencies节点中

#### 如何查找始都为项目开发时候要用到的

https://www.npmjs.com/

在install里面找

install with npm目录下

### npm包

|            |                          |
| ---------- | ------------------------ |
| i5ting_toc | md文档转换为html的小工具 |
|            |                          |
|            |                          |
|            |                          |
|            |                          |
|            |                          |
|            |                          |
|            |                          |
|            |                          |

## 开发自己的包

```json
{
  "name": "itheima-tools-ygt",
  "version": "1.0.0",
  "description": "提供了格式化时间，HTMLEscape相关功能",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "author": "",
  "license": "ISC",
    //提供关键字
  "keywords": [
    "itheima",
    "dataForamt",
    "escape"
  ]
}

```

#### README.md文件

包含

- 安装方式
- 导入方式
- 格式化方式
- 转移HTML中的特殊字符
- 远远HTML中的特殊字符
- 开源协议

### 发布包

1. 终端执行 `npm login` 命令，依次输入用户名密码邮箱

   - 在运行之前，必须先把下宝的服务器地址切换为**<u>npm的官方服务器</u>**，否则会导致发布包失败！

### 删除包

   - `npm unpublish` 包名 --force命令，即可从npm删除已发布的包
   - `npm unpublish` 命令只能删除72小时以内发布的包
   - `npm unpublish` 删除的包，在24小时内不允许重复发布

### 加载机制

#### 模块的加载机制

内置模块是由 Node.js 官方提供的模块，内置模块的加载优先级最高

例如，require('fs')始终返回额是fs的模块。即使在node_modules 目录下有名字相同的包也叫做fs

#### 自定义模块的加载机制

- 使用 `require()` 加载自定义模块时，必须指定以 ./ 或 ../ 开头的路径标识符。在加载自定义模块时，如果没有指定 ./ 或 ../这样的路径标识符，则 node 会把它当作内置模块或者第三方模块进行加载

- 同时，在使用 require() 导入自定义模块时，如果省略了文件的扩展名，则Node.js 会按顺序分别尝试加载以下的文件：
  - 按照确切的文件名进行加载
  - 补全 .js 扩展名进行加载
  - 补全 .json 扩展名进行加载
  - 补全 .node 扩展名进行加载
  - 加载失败，最终报错

#### 第三方模块的加载机制

如果传递给require() 的模块标识符不是一个内置模块，也没有以 ‘./’ 或 '../' 开头，则Node.js 会从当前模块的父目录开始，尝试从 /node_modules文件夹中加载第三方模块。

如何查看查找顺寻：`console.log(module)` 中的path

如：

```js
paths: [
    'd:\\youguangtao gitee文件备份\\node-learn\\06.开发自己的包\\node_modules',
    'd:\\youguangtao gitee文件备份\\node-learn\\node_modules',
    'd:\\youguangtao gitee文件备份\\node_modules',
    'd:\\node_modules'
  ]
```

### 目录作为模块

把当前目录作为模块标识符，传递给 require()进行加载的时候，有三种加载方式

1. 在被加载的目录下查找一个叫做package.json 的文件，并寻找main属性，作为require() 加载的入口
2. 如果目录里没有 package.json 文件，或者mai入口不存在或无法解析，则Node.js 将会是图加载目录下的 index.js 文件。
3. 如果以上两步都失败了，则Node,js会在中断打印错误信息，报告模块的确实： Error: Cannot find module 'xxx'

## Express

[中文官网](https://www.expressjs.com.cn/)

```js

const express = require('express')

const app = express()

// 调用 app.listen(端口号，启动成功后的回调函数)，启动服务器
app.listen(80,()=>{
  console.log('express server running at http://127.0.0.1')
})

//这里的:id是一个动态的参数
app.get('/user/:id' , (req,res) => {
  console.log(req)
  res.send({name:'zs',age:20,gender:'男',id:req.params.id})
})
  
app.post('/user',(req,res)=>{
  res.send('请求成功')
})

// 通过 req.query 可以获取到客户端发送过来的查询参数
// 注意：默认情况下，req.query 是一个空对象
app.get('/',(req,res)=>{
  res.send(req.query)
})
```

### express.static() —— 托管静态资源	

通过 `express.static()` 我们可以非常方便的创建一个静态资源服务器 

```js
app.user(express.static('public'))
```

就可以通过访问public目录中所有的文件

如：http://localhost:3000/images/gj/jpg

- Express在指定的静态文件中查找文件，并对外提供资源的访问路径，因此，<u>存放静态文件的目录名不会出现在URL中</u>

如

```js
//调用express.static()方法，快速的对外提供静态资源
const app = express('./clock')
```

#### 托管多个资源目录

访问静态资源文件时，express.static() 函数会根据目录的添加顺序查找所需的文件

```js
app.user(express.static('public'))
app.user(express.static('files'))
```

访问静态文件时，express.static()函数会根据目录的添加顺序查找所需的文件

#### 挂载路径前缀

如果希望在托管的静态资源访问路径之前，挂载路径前缀，可以使用以下方法

```js
app.use('/public' , express.static('public'))
```

### nodemon

用于监听项目文件变动，帮我们自动重启项目

#### 安装nodemon

```js
npm install -g nodemon
```

#### 使用nodemon

用来代替 `node app.js`

```
nodemon app.js
```

### Express 路由

在Express中，路由指的是客户端的请求与服务器处理函数之间的映射关系

Express中的路由由3分钟组成，分别是请求的类型，请求的URL地址，处理函数，格式如下

```
app.METHOD(PATH,HANDLER)

app.get('/nihao',()=>{})
```

每当一个请求到达服务器后，需要先经过路由的匹配，只有匹配过后，才会调用对应的处理函数

如果请求类型和请求的URL同时匹配成功，则Express会将着此请求，转交给对应的function函数进行处理

注意

- 按照先后顺序进行匹配
- 请求类型和请求的URL同时匹配成功，才会调用对应的对应函数

### 模块路由

为了方便对路由进行模块化管理，推荐将路由抽取成单独的模块

```js

const express = require('express')
const router = express.Router()

//挂载路由对象
router.get('/user/list',(req,res)=>{
  res.send('Get user list')
})

router.post('/user/add',(req,res)=>{
  res.send('Add new user')
})

module.exports = router
```

#### 注册导入路由模块

```
//导入路由模块
const userRouter = require('./router/user.js')

//使用app.use()注册路由模块
app.use(userRouter)
```

注意，app.use() 函数的作用，就时用来注册全局中间件

### 中间件一般注册在所有路由之前，错误路由记在所有中间件之后

### 中间件，(全局)

```js
const express = require('express')
const app = express()

const mw = function (req,res,next) {
  req.startTime = time
  console.log('这是最简单的中间件函数')
  // 把流转关系，转交给下一个中间件或路由
  next()
  }

  app.use(mw)

  app.listen(81,(req,res)=>{
    console.log('http://127.0.0.1')
  })

  app.get('/',(req,res)=>{
    console.log('调用了/这个路由')
  })

  app.get('/user',(req,res)=>{
    console.log('调用了/user这个路由')
  })

```

### 中间件，(局部)

```js
const express = require('express')
const app = express()

app.listen(80)

const mwl = (req,res,next)=>{
  console.log('调用了局部生效的中间件')
  next()
}
const mw2 = (req,res,next)=>{
  console.log('调用了局部生效的中间件2')
  next()
}

app.get('/user',mwl,mw2,(req,res)=>{
  console.log(req)
  res.send('Home page.')
})

app.get('/',(req,res)=>{
  res.send('User page.')
})
```

### 中间件分类

1. 应用级别的中间件

   - 通过app.use()和app.get()或app.post() , 绑定到app实例上的中间件，叫做应用级别的中间件，代码示例如下

2. 路由级别的中间件

   - 绑定到express.Router()实例上的中间件，叫做路由级别的中间件，它的用法和应用级别中间件没有任何区别，只不过，应用级别中间件是绑定到app实例上，而路由级别中间件绑定到router实例上，代码示例如下

     ```js
     var app = express()
     var router = express.Router()
     
     router.use(function(req,res,next){
     	console.log('time',date.now())
     	next()
     })
     
     app.use('/',router)
     ```

     

3. 错误级别的中间件
   中间件一般注册在所有路由之前，错误路由记在所有中间件之后

   ```js
   const express = require('express')
   const app = express()
   
   app.listen(80)
   
   const mwl = (err,req,res,next)=>{
     console.log('发生了错误'+err.message)
     res.send('Error!'+err.message)
   }
   //
   app.get('/',(req,res)=>{
     throw new Error('服务器内部发生了故障')
     res.send('Home page.')
   })
   // 注意app.get 和app.use的顺序
   app.use( mwl)
   ```

   

4. Express内置的中间件

   - express.static 快速托管静态资源的内置中间件，例如HTML文件，图片，CSS样式

   - express.json 解析JSON格式的请求体数据，(有兼容性 4.16.0+版本使用)

     ```js
     // 配置解析 application/json 格式数据的内置中间件
     app.use(express.json())
     ```

     

   - express.unlencoded 解析URL-encoded解析URL-encoded格式的请求体数据，(有兼容性，仅在4.16.0+版本使用)

     ```js
     // 配置解析 application/x-www-from-url-urlencode 格式数据的内置中间件
     app.use(express.urlencoded({extended: false}))
     ```

     

5. 第三方的中间件
   如 body-parser
   注意，Express内置的 express.urlencoded中间件，就是基于body-parser这个第三方中间件进一步封装出来的

   ```js
   const express = require('express')
   const app = express()
   
   // 使用外置中间件进行运用
   const parser = require('body-parser')
   
   app.use(parser.urlencoded({extended:false}))
   
     app.listen(80,(req,res)=>{
       console.log('http://127.0.0.1')
     })
   
   
     app.get('/user',(req,res)=>{
       console.log(req.body)
       res.send("received")
     })
   
   ```

   

### 自定义中间件

```js
const express = require('express')
const app = express()

app.listen(80)

// 解析表单数据的中间件
const mwl = (req,res,next)=>{
  // 定义一个str字符串，专门用来存储客户端发送过来的请求提数据 
  let str = ''
  // 监听req的data事件 (客户端发送过来的心的请求体数据)
  req.on('data',(chunk)=>{
    //拼接请求体数据，隐式转换为字符串
    console.log(chunk)
    str += chunk 
  })

  // 监听end事件,只要出发了，就可以接收完毕了 
  req.on('end',()=>{
    // 在str中存放的是完整的请求体数据
    console.log(str)
    // 把字符串格式的请求提数据，解析成对象格式
  })
  next()
}

app.use(mwl)

app.post('/user',(req,res)=>{
  res.send('ok')
})
```

#### 利用querystring 模块解析请求体数据

Node.js 内置了querystring模块，专门用来处理查询字符串，通过这个模块提供的parse()函数，可以轻松把查询字符串解析成对象的格式

## 使用Express写接口

### CORS跨域资源共享

Cross-Origion Resource Share

```
npm install cors
//一定要再路由之前，配置cors这个中间件，从而解决接口跨域的问题
const cors = require('cors')
app.use(cors())
```

#### Access-Control-Allow-Origin

origin 参数的值制定了允许访问该资源的外域URL,

下面字段只允许来自`http://itcast.cn`的请求

如 `res.setHeader('Access-Control-Allow-Origin','http://itcast.cn')`

如果设置为通配符 `*`,则表示允许来自任何域的请求

如 `res.setHeader('Access-Control-Allow-Origin','*')`

#### Access-Control-Allow-Headers

默认情况下，仅支持客户端向服务端发送如下的9个请求头

简单首部，如 [simple headers](https://developer.mozilla.org/zh-CN/docs/Glossary/Simple_header)、[`Accept`](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Headers/Accept)、[`Accept-Language`](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Headers/Accept-Language)、[`Content-Language`](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Headers/Content-Language)、[`Content-Type`](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Headers/Content-Type)、DPR，Downlink、Save-Data、Viewport-Width（只限于解析后的值为 `application/x-www-form-urlencoded`、`multipart/form-data` 或 `text/plain` 三种 MIME 类型（不包括参数）），它们始终是被支持的，不需要在这个首部特意列出。

*如果客户端向服务器发送了额外的请求头消息，那么需要再服务器端，通过Headers对额外的请求头进行声明，否则请求会失败*

#### Access-Control-Allow-Methods

通常情况下，CORS仅支持客户端发起GET，POST，HEAD请求

如果希望通过PUT，DELETE等方式请求服务器的资源，那么需要通过methods来致命实际请求所允许使用的HTTP方法，如
 `res.setHeader('Access-Control-Allow-Origin','POST, GET, DELETE, HEAD')`
 `res.setHeader('Access-Control-Allow-Origin','*')`

#### 预检请求

只要符合以下任何一个条件的请求，都需要进行预请：

- 请求方式为GET，POST，HEAD 之**<u>外</u>**的请求的Method类型
- 请求头中包含自定义头部字段
- 向服务器发送了 application/json 格式的数据

*在浏览器与服务器正式通信之前，浏览器会发送OPTION请求进行预检，以获知服务器是否允许该实际请求，所以这一次OPTION请求称为“预检请求”。服务器成功响应预检请求后，才会发送真正的请求，并且携带真实数据*

简单请求和预检请求的区别

简单请求的特点：客户端与服务器之键之会发生一次请求

预检请求的特点：客户端与服务器之键会发生两次请求，OPTION预检请求成功之后，才会真正发起真正的请求

### JSONP接口

回顾JSONP的概念与特点

概念： 浏览器端通过`<script> `标签的src属性，请求服务器上的数据，同时，服务器返回一个函数的调用。这种请求数据的方式叫做JSONP

特点：

1. JSONP 不属于真正的Ajax，因为它没有使用XMLHttpRequest这个对象
2. JSONP仅支持GET请求，不知处POST，put，DELETE等请求

创建JSONP接口的注意事项

如果项目中已经配置了CORS跨域资源攻向，为了防止冲突，必须在配置CORS中间件之前声明JSONP的接口。否则JSONP接口会被处理成开启了CORS的接口。实例代码如下

```js
//优先创建JSONP接口，这个接口不会被处理成CORS接口
app.get('/api/jsonp',(req,res)=>{
    
})
//在配置CORS中间件，后续所有接口，都会被处理成CORS接口
app.use(cors())

//这是一个开启了CORS的接口
app.get('api/get',(req,res)=>{})
```

JSONP具体代码

```js
app.get('/api/jsonp'.(req,res)=>{
	// 获取客户端发送过来的回调函数的名字
    const funcName = req.query.callback
    //要得到JSONP形式发送给客户端的数据
    const data = {name:'zs',age:22}
    //根据前两部得到的数据，拼接处一个函数调用的字符串
    
})
```

网页中使用jQuery发起JSONP请求

```js
$('#btnJSONP').on('click',function(){
    $.ajax({
        methed:'GET',
        url:'http://127.0.0.1/api/jsonp',
        dataType:'jsonp'//表示要发起JSONP请求
        success:function(res){
        console.log(res)
    }
    })
})
```

## 数据库基本概念

常见数据库

- MySQL（都有，目前使用最多）
- Oracle（收费）
- SQL Server（收费）
- Mongodb（都有，目前使用最多）

其中MySQL，Oracle，SQL Server属于传统型数据库，又叫关系型数据库，用法类似

而Mongodb属于新型数据库，又叫非关系型数据库，NoSQL数据库，一定程度上姆布勒数据库的缺陷

### 传统型数据库数据组织结构

分工作簿（数据库），工作表（数据表），数据行，列

database，table，row，field

### 安装配置mysql模块

`npm install mysql`

### 配置Mysql

```js
// 导入mysql模块
const mysql = require('mysql')
// 建立与Mysql数据库的连接
const db = mysql.createPool({
	host:'127.0.0.1',//数据库IP地址
    user:'root',//登录数据库的账号
    password:'admin123',//登录数据库的密码
    database:'my_db_01'//指定要操作那个数据库
})
```

### 或者，配置[MySQL2](https://sidorares.github.io/node-mysql2/zh-CN/docs#%E7%BB%93%E6%9E%9C%E8%BF%94%E5%9B%9E)

```js
// 导入模块
const mysql = require("mysql2");

// 创建一个数据库连接
const connection = mysql.createConnection({
  host: "127.0.0.1",
  user: "root",
  database: "wzvtc_user_system",
  password: "123456",
});

// 简单查询
connection.query("SELECT * FROM `area`", function (err, results, fields) {
  console.log(results); // 结果集
  console.log(fields); // 额外的元数据（如果有的话）
  console.log(err);
});
```

## Web开发模式

### 服务端渲染的Web开发模式

服务端渲染的概念：服务器发送给客户端的HTML页面，是在服务器通过字符串的拼接，动态生成的。因此，客户端不需要使用Ajax这样的技术额外请求页面的数据。

```
app.get('./index.html',(req,res)=>{
//要渲染的数据
const user = {name :'zs',age:20}
//服务器通过字符串的拼接，动态生成HTML内容
const html = '<h1>姓名：${user.name},年龄：${user.age}</h1>'
//把生成好的页面内容相应给客户端，因此，客户端拿到的是带有真实数据的HTML页面
res.send(html)
})
```

#### 优缺点

有点

1. 前端耗时少
2. 有利于SEO

缺点

1. 占用服务器端资源
2. 不利于前后端份力，开发效率低，无法进行分工合作

### 前后端分离的Web 开发模式

1. 便于分工合作
2. 用户体验好
3. 减轻了服务端渲染压力

缺点

1. 不利于SELP，利于SSR(server side render)

## Express中Session

### 配置express-session中间件

```js
// 导入session中间件
var session = require('express-session')

// 配置Session中间件
app.use(session({
    secret:'keyboard_cat',
    resave:false,
    saveUninitialized:true
}))


```

## JWT(token)

`npm install jsonwebtoken express-jwt`

jsonwebtoken 用于生成JWT字符串

express-jwt用于将JWT字符串解析还原成JSON对象

#### 定义JWT密钥

```js
const secretKey = 'miyao'
var { expressjwt: jwt } = require("express-jwt");

app.get(
  "/protected",
  jwt({ secret: "helloworld", algorithms: ["HS256"] }),
  function (req, res) {
    if (!req.auth.admin) return res.sendStatus(401);
    res.sendStatus(200);
  }
);
```

Synchronous Sign with default (HMAC SHA256)

```js
var jwt = require('jsonwebtoken');
var token = jwt.sign({ foo: 'bar' }, 'shhhhh');
```

Synchronous Sign with RSA SHA256

```js
// sign with RSA SHA256
var privateKey = fs.readFileSync('private.key');
var token = jwt.sign({ foo: 'bar' }, privateKey, { algorithm: 'RS256' });
```

#### 把JWT字符串还原为JSON对象

客户端每次在访问那些有权限接口的时候，都需要主动通过请求头中的Authorization字段，将Token字符串发送到服务器进行身份认证

此时，服务器可以通过express-jwt这个中间件，自动将客户端发送过来的Token解析还原成JSON对象

```js
// 使用app.use()来注册中间件
// expressJWT({secret:secretKey}) 就是用来解析Token的中间件
//.unless({path:[/^\/api\//]}) 用来指定那些接口不需要访问权限
// 注意，只要配置成功了express-jwt这个中间件，就可以把解析出来的用户信息，挂载到req.user属性上
app.use(expressJWT({secret:secretKey}).unless({path:[/^\/api\//]}))
```

捕获解析JWT失败后产生的错误

```js
app.use((err,req,res,next)=>{
	if(err.name === 'UnauthorizedError'){
		return res.send({status:401,message:'无效地token'})
	}
    //其他原因导致的错误
    res.send({status:500,message:'未知错误'})
})

```

