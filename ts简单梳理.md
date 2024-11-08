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

##### ATT

当使用**类型声明**限制函数返回值为`void`时，TS并不会严格要求函数返回空

```typescript
type LogFunc = () => void;
// 此时不会报错
const f1: LogFunc = () =>{
	return 66
}
```

why?

为了确保以下代码成里，我们知道 `Array.prototype.push`的返回一个数字，而`Array.prototype.forEach`方法是期望其回调的返回值是`void`

```typescript
const src = [1,2,3];
const dst = [0];
// 箭头函数只有一句的且简写的时候，会把值自动作为return的值，比如push会返回长度，那么这里就会return number
src.forEach((el) => dst.push(el))
```



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
//{0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', Up: 0, Down: 1, Left: 2, Right: 3}
```

##### 常量枚举 - 数字内联

常量枚举是一种特殊的枚举类型，它使用const关键字定义，**在编译时会被内联，避免生成一些额外的代码**

`内联 指TS在编译时，把枚举成员引用替换为他们的实际值，而不是生成额外的枚举对象，这样可以减少生成的代码量，并且提高性能。`

#### type

`  | and & `

type可以为任意类型创建别名

##### 基本用法

```typescript
type num = number;

let price: num;
price = 100;
```

##### 联合类型

```typescript
type Status = number | string
type Gender = '男' | '女'

function printStatus(status: Status){
    console.log(status);
}

function logGender(str: Gender){
    console.log(str)
}
```

##### 交叉类型

其实就是`并区间`的概念

如：

```typescript
type Status = number | string
type Gender = '男' | '女'
// 此时aa将只能是'男' | '女'
type aa = Status & Gender;

```

```typescript
type Area = {
    height: number;
    width: number;
}

type Address = {
    num: number;
    cell: number;
    room: string;
}

type House = Area & Address

const house:House={
    height:100,
    width:100,
    num:3,
    cell:4,
    room:'702'
} 
```

# 类

和java一样，可实现多个接口，用法也和java一样

## 封装

```typescript
// 定义
class Person{
  name:string;
  age: number;
  constructor(name:string,age:number){
    this.name = name;
    this.age = age;
  }
  speak():string{
    return this.name
  }
}

const p1 = new Person('张三',18)
p1.speak
```

## 继承

```typescript
// 继承
class Student extends Person{
  grade:string;
  constructor(name:string,age:number,grade:string){
    super(name,age)
    this.grade = grade;
  }
  // 如果打算覆写最好加 override
  override speak(): string {
      return "stu"
  }
  study():string{
    return `${this.name}are learning in ${this.grade}`
  }
}
```

## 属性修饰符

| 修饰符    | 含义     | 具体规则                            |
| --------- | -------- | ----------------------------------- |
| public    | 公开的   | 类内部，子类，类外部使用            |
| protected | 受保护的 | 类内部，子类使用                    |
| private   | 私有的   | 类内部使用                          |
| readonly  | 只读属性 | 属性无法更改，只能在construct中使用 |

## 抽象+继承

`  constructor(public weight: number) {}直接默认生成了，可调`

```typescript
// 定义
abstract class Package {
  constructor(public weight: number) {}
  abstract calculate(x: number, y: number): number;
  printPackage() {
    console.log(`包裹重量为${this.weight}`);
  }
}
class StandardPackage extends Package {
  constructor(weight: number, public unitPrice: number) {
    super(weight);
  }
  calculate(x: number, y: number): number {
    return this.weight * this.unitPrice;
  }
}
```

## 接口

拓展第三方库类型

**和java中不一样，java不能继承，只能实现**

### 定义类的结构

```typescript
interface PersonInterface{
  name: string
  age: number
  speak(n:number):void
}
class Person implements PersonInterface{
  constructor(
    public name: string,
    public age: number,
  ){}
  speak(n: number): void {
    for(let i = 0;i < n;i++){
      console.log(`${this.name}`)
    }
  }
}
```

### 定义对象的结构

为了内容完整性了

```typescript
interface UserInterface{
  name: string
  readonly gender:string
  age?: number
  run:(n:number) => void
}

const user: UserInterface = {
  name: "张三",
  gender: "男",
  run: function (n: number): void {
    console.log(`${this.name} run ${n} meter`)
  }
}
```

### 定义函数的结构

```typescript
interface CountInterface{
	(a:number, b:number):number;
}

const count:CountInterface = (x,y)=>{
	return x+y
}
```



### 接口之间的继承

```typescript
interface PersonInterface{
  name: string
  age?: number
}

interface StudentInterface extends PersonInterface {
  grade:string
}

const stu:StudentInterface ={ 
  name:"张三",
  grade:"一班"
}
```



### 接口的自动合并

一般用于拓展第三方库的类型

```typescript
interface PersonInterface{
  name: string
  age?: number
}

interface PersonInterface {
  grade:string
}

const stu:PersonInterface ={ 
  name:"张三",
  grade:"一班"
}
```

# 泛型

`泛型允许我们在定义函数，类，或接口的时候，使用类型参数来表示未指定的类型，这些参数在具体使用时，才被直顶具体的类型，泛型能让同一段代码适用于多种类型，同时仍然保持类型的安全`

### 泛型函数

```typescript
function logData<T>(data: T): T{
  console.log(data)
  return data
}

logData<number>(100);
logData<string>('hello')
```

### 泛型可以有多个

```typescript
function logData2<T,U>(data1: T,data2:U): T | U{
  console.log(data1,data2)
  return data2
}

logData2<number,boolean>(100,true);
logData2<string,number>('hello',666);
```

### 泛型接口

```typescript
interface PersonInterface<T>{
	name:string,
	age:number,
	anyInfo:T
}
// 使用
type GradeInfo = {
    grade:string;
    school:string;
}
let p:PersonInterface<UserInfo> = {
	name:'tom',
    age:18,
    GradeInfo:{
        grade :'123';
    	school:'456';
    }
}

```

# 类型说明文件

`类型声明文件是TypeScript的一种特殊文件，通常以.d.ts作为扩展名。它的主要作用是为现有的JavaScript代码提供类型信息，使得TypeScript能够在使用这些JavaScript库时进行类型检查和提示`

通常放到@types文件中

例子：

demo.js

```js
export function add(a,b){
    return a+b
}
```

demo.d.ts

```typescript
declate function add(a:number,b:number):number;
 
export {add};
```

# 装饰器

扩展类

### 类装饰器(无法传参)

```typescript
function fun(target:any){
	target.prototype.username = '张三'
}

@fun
class Person{
    
}
let p1 = new Person();
console.log(p1.username); // 张三
```

### 装饰器工厂

人话就是返回一个方法，让方法调用

```typescript
function fun1(options: any){
	return (target:any)=>{
        target.username = options.name;
        target.prototype.age = options.age;
    }
}

@fun1({
    name:'李四',
    age:18
})
class Obj1{
    
}

let obj1 = new Obj1();
console.log(Obj1.userName, obj1.name , obj1.age)
```

### 装饰器组合

自上而下先获取到所有的真正的装饰器，然后下到上执行

```typescript
function demo1( target:any ){
    console.log('demo1')
}
function demo2(  ){
    console.log('demo2')
    return ( target:any )=>{
        console.log('demo2里面')
    }
}
function demo3( ){
    console.log('demo3')
    return ( target:any )=>{
        console.log('demo3里面')
    }
}
function demo4( target:any ){
    console.log('demo4')
}

@demo1
@demo2()
@demo3()
@demo4
class Person{

}

/*结果是：
demo2
demo3
demo4
demo3里面
demo2里面
demo1
*/
```

### 属性装饰器

```typescript
function fun3( arg:any ){
    return ( target:any , attr:any )=>{
        target[attr] = arg;
    }
}

class Obj3{
    
    @fun3('张三')
    userName:string
    
}   
let obj3 = new Obj3();
console.log( obj3.userName );
```

### 方法装饰器

```typescript
function test(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  console.log(target);
  console.log(propertyKey);
  console.log(descriptor);
  // return descriptor;
}

class Person {

  @test
  sayName() :string{
    console.log('say name...');
    return 'say name';
  }
}

let p = new Person();
p.sayName();

//最终输出
/*
{sayName: ƒ}
sayName
{writable: true, enumerable: false, configurable: true, value: ƒ}
say name...
*/
```

