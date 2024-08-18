### js和TS特殊的区别

- js是在运行前进行检查，ts实在编译的时候就进行了检查

### TS转成js

#### 命令行编译

`tsc index.ts`就可以转成js文件

#### 自动化编译

1. `tsc --init` , 创建ts编译文件，会自动生成 `tsconfig.json`配置文件，默认编译js版本为ES7,可手动调整为其他版本
2. `tsc --watch`
   1. 当编译出错的时候不生成 `.js`文件 -> ，也可以修改成 `tsconfig.json`中的noEmitOnError 配置

# 类型

## 声明

```typescript
let a: string
let b: string
let c: number = -99 // 等于 let c = -99
// @param x,y is number
// return number
function count(x:number,y:number):number{
    return x + y;
}
```

## 总览

| js        | ts                                        |
| --------- | ----------------------------------------- |
| string    | js的所有                                  |
| number    | any 任意                                  |
| boolean   | unknown 未知                              |
| null      | never                                     |
| undefined | void                                      |
| bigint    | tuple                                     |
| symbol    | enum 枚举                                 |
| object    | 两个自定义类型的方式<br />type，interface |

#### string

```typescript
let str : string //ts官方推荐写法
// let str : String
str = 'hellp'
str = new String('hello')//报错，不能将类型String分配给 string
```

string是基元，String是包装器对象，

简而言之，大写的内置构造函数，很少使用



#### unknown

```typescript
let a: unknown;

a = false
a = 'hello'

let x: string

// 可以这么用，判断
if(typeof a === 'string'){
x = a;
}

// 断言
// 第一种
x = a as string
// 第二种
x = <string>a
```

unknown会<u>强制开发者在使用之前进行类型检查</u>，从而提供更强的类型安全性

读取any类型数据的任何属性都不会报错，而unknown正好与之相反

#### never

啥也不是，undefined，null，''，0都不行

- 注意，返回never的函数不能具有可访问的终结点

```typescript
function demo():never{}
// 这里只有两种可能，要么这个方法死循环，要么这个方法无法正常结束
// 一旦执行，那么就直接报错，程序访问异常
```

- never一般是ts主动推断出来的
- never也可以用于限制函数的返回值，

```typescript
// 限制throwError函数不需要有任何返回值，任何值都不行，像undefined，null都不行
function throwError(str: string): never{
	throw new Error('程序异常退出' + str)
}
```

#### void

java一样

#### object

##### object

- object

存的是非原始类型数据

可以这么声明

```typescript
let person : {
    name: string , //分隔符可用逗号，分号甚至换行，
    age?: number , //问号代表可有可无
    [key:string]: any //索引签名，person可以有任意一个key，不过这里的key可以为随意字母如index，只要保证key是string类型
}
```



- ###### Object

存的类型是可以调用到Object方法的类型

人话，除了null，undefined

##### function

```typescript
let count:(a:number,b:number) => number //这里的箭头是做形式分隔，不是箭头函数，意思是return number 

count = function(c, d){
    return c + d;
}//然后就可以直接这么写，将直接默认param，return里面的类型是已设定的类型
```

##### array

```typescript
let arr: Arrary<number>;
```

#### tuple

元组(Tuple)是一种特殊的数组类型，可以存储固定数量的元素，并且每个元素的类型是已知的且可以不同，元组用于精确描述一组值的类型，？可以表示可选元素

```typescript
// 第一个元素必须是string类型，第二个元素必须是number类型
let arr1:[string,number];

// 第一个元素类型必须是number类型，第二个元素是可选的，如果存在，必须是boolean类型
let arr2:[number,boolean?];

// 第一个元素必须是number类型，后面的元素可以是任意数量的string类型
let arr3:[number,...string[]]
```

#### enum

##### 数字枚举

数字枚举是一种最常见的枚举类型，其成员的值会自动递增，且数字枚举还具备反向映射的特点，在下面代码的打印中，不难发现：可以通过值来获取对应枚举成员的名称。

```typescript
enum Direction{
 Up,
 Down,
 Left,
 Right
}

console.log(Direction)

```

