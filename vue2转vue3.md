# vue2转vue3

## API：setup()(组件生命周期)

新的 `setup` 选项是在组件创建**之前**, `props` 被解析之后执行，是组合式 API 的入口。

> 在添加了setup的script标签中，组件只需引入**不用注册**，属性和方法也**不用返回**，也**不用写`setup`函数**，也**不用写`export default`**，甚至是[自定义指令](https://so.csdn.net/so/search?q=自定义指令&spm=1001.2101.3001.7020)也可以在我们的`template`中自动获得。

tip：在 `setup` 中应该**避免使用 `this`**，因为它不会找到组件实例。

```vue
<script setup>
...
</script>
```

#### 组件生命周期

| Vue 2 生命周期 | Vue 3 生命周期      | 执行时间说明                               |
| -------------- | ------------------- | ------------------------------------------ |
| beforeCreate   | **setup**           | 组件创建前执行                             |
| created        | **setup**           | 组件创建后执行                             |
| beforeMount    | **on**BeforeMount   | 组件挂载到节点上之前执行                   |
| mounted        | **on**Mounted       | 组件挂载完成后执行                         |
| beforeUpdate   | **on**BeforeUpdate  | 组件更新之前执行                           |
| updated        | **on**Updated       | 组件更新完成之后执行                       |
| beforeDestroy  | **on**BeforeUnmount | 组件卸载之前执行                           |
| destroyed      | **on**Unmounted     | 组件卸载完成后执行                         |
| errorCaptured  | **on**ErrorCaptured | 当捕获一个来自子孙组件的异常时激活钩子函数 |

## defineProps 和 defineEmits.，父子组件传递参数

### defineProps ----> [用来接收父组件传来的 props]

通过`defineProps`指定当前 props 类型，获得上下文的props对象。

#### 父组件传递参数：

```vue
<template>
  <div class="box">
    <test-com :info="msg" time="42分钟"></test-com>
  </div>
</template>
<script lang="ts" setup>
import TestCom from "../components/TestCom.vue"
let msg='今天是2023年3月14日'
</script>
```

#### 子组件接受参数：

```vue
<template>
    <div>
        <h2> 啦啦啦啦啦啦啦啦</h2>
        <p>信息: {{ info}}</p>
        <p> {{ time }}</p>
    </div>
</template>
<script lang="ts" setup>
import {defineProps} from 'vue'
defineProps({
    info:{
        type:String
        default:'----'
    },
    time:{
        type:String,
        default:'0分钟'
    },
})
</script>
```

#### 解决了父组件向子组件传值，那么子组件怎么向父组件抛出事件？

### `defineEmits`

#### 定义 emit

defineEmit ----> [子组件向父组件事件传递]

使用`defineEmit`定义当前组件含有的事件，并通过返回的上下文去执行 emit。

**代码示列**：

```vue
<script setup>
  import { defineEmits } from 'vue'
  const emit = defineEmits(['change', 'delete'])
</script>
```

#### 父子组件通信

**defineProps** 用来接收父组件传来的 props ; **defineEmits** 用来声明触发的事件。

父组件

```vue
//父组件
<template>
    //监听子组件的getChild方法，传msg给子组件
    <Child @getChild="getChild" :title="msg" />
</template>
<script setup>
import { ref } from 'vue'
import Child from '@/components/Child.vue'
const msg = ref('parent value')
const getChild = (e) => {
    // 接收父组件传递过来的数据
    console.log(e); // child value
}
</script>
```

 子组件

```vue
//子组件
<template>
    <div @click="toEmits">Child Components</div>
</template>
 
<script setup>
// defineEmits,defineProps无需导入，直接使用
const emits = defineEmits(['getChild']);
//接收父组件传来的props
const props = defineProps({
    title: {
        type: String,
        defaule: 'defaule title'
    }
});
const toEmits = () => {
    // 向父组件抛出带参事件getChild（其中参数是child value）
    emits('getChild', 'child value') 
}
// 获取父组件传递过来的数据
console.log(props.title); // parent value
</script>
```

- 子组件通过 defineProps 接收父组件传过来的数据
- 子组件通过 defineEmits 定义事件发送信息给父组件

## 组件自动注册

在 script setup 中，引入的组件可以直接使用，无需再通过components进行注册，并且无法指定当前组件的名字，它会自动以文件名为主，也就是不用再写name属性了。

**示例**：

```vue
<template>
  <div class="home">
    <test-com></test-com>
  </div>
</template>
 
<script lang="ts" setup>
 
// 组件命名采用的是大驼峰，引入后不需要在注册
// 在使用的使用直接是小写和横杠的方式连接 test-com
import TestCom from "../components/TestCom.vue"
 
</script>
```

## useSlots() 和 useAttrs()

获取slots,attrs

- userAttrs : 这里是用来获取attrs数据，但是这和vue2不同，里面包含了class,属性,方法

  ```vue
  <template>
  	<component v-bind = 'attrs'></component>
  </template>
  <script setup lang='ts'>
      const attrs = useAttrs();
  </script>    
  ```

- `useSlots`: 顾名思义，获取插槽数据。

  ```vue
  // 旧
  <script setup>
    import { useContext } from 'vue'
    const { slots, attrs } = useContext()
  </script>
   
  // 新
  <script setup>
    import { useAttrs, useSlots } from 'vue'
    const attrs = useAttrs()
    const slots = useSlots()
  </script>
  ```

  

## 向组件暴露出自己的属性 defineExposeAPI

在传统的写法中，我们可以通过父组件中的ref实例的方式去访问子组件的内容，但在script setup 中，该方法就不能用了，setup相当于是一个闭包，除了内部的template模板，谁都不能访问内部的数据和方法

> <script setup> 的组件默认不会对外部暴露任何内部声明的属性。

如果需要对外暴露setup 中的数据和方法，需要使用defineExposeAPI

- defienExpose 不需要导入，可以直接使用

示例：

子组件

```vue

//子组件
<template>
    {{msg}}
</template>

<script setup>
import { ref } from 'vue'
let msg = ref("Child Components");
let num = ref(123);
defineExpose({
    msg,
    num
});
</script>
```

父组件

```vue
//父组件
<template>
    <Child ref="child" />
</template>
 
<script setup>
import { ref, onMounted } from 'vue'
import Child from '@/components/Child.vue'
let child = ref(null);
onMounted(() => {
    console.log(child.value.msg); // Child Components
    console.log(child.value.num); // 123
})
</script>
```

## 新增指令 v-memo

`v-memo`会记住一个模板的子树，元素和组件上都可以使用。

该指令接收一个固定长度的数组作为依赖值进行“记忆比对”。如果数组中的每个值都和上次渲染的时候相同，则整个子树的更新会被跳过。

即使是虚拟 DOM 的 VNode 创建也将被跳过，因为子树的记忆副本可以被重用。

-> 使用后渲染的速度会非常地快

正确的声明记忆数组是很重要的

开发者有责任指定正确的依赖数组，以避免必要的更新被跳过。

如

```vue
<li v-for="item in listArr"  :key="item.id"  v-memo="['valueA'，'valueB']">
    {{ item.name   }}
</li>
```

`v-memod`的指令使用较少，它的作用是：缓存模板中的一部分数据。

只创建一次，以后就不会再更新了。也就是说用内存换取时间。

## style v-bind

```vue

<template>
  <span> 啦啦啦啦啦啦啦啦啦啦 </span>  
</template>
<script setup>
  import { reactive } from 'vue'
  const state = reactive({
    color: 'red'
  })
</script>
<style scoped>
  span {
    /* 使用v-bind绑定state中的变量 */
    color: v-bind('state.color');
  }  
</style>
```

## 定义组件的其他配置

配置项的缺失，有时候我们需要更改组件选项，在`setup`中我们目前是无法做到的。我们需要在`上方`再引入一个 `script`，在上方写入对应的 `export`即可，需要单开一个 script。

<script setup> 可以和普通的 <script> 一起使用。

普通的 `<script>` 在有这些需要的情况下或许会被使用到：

- 无法在 `<script setup>` 声明的选项，例如 `inheritAttrs` 或通过插件启用的自定义的选项。
- 声明命名导出。
- 运行副作用或者创建只需要执行一次的对象。

在script setup 外使用export default，其内容会被处理后放入原组件声明字段。

```vue

<script>
// 普通 `<script>`, 在模块范围下执行(只执行一次)
runSideEffectOnce()
// 声明额外的选项
  export default {
    name: "MyComponent",
    inheritAttrs: false,
    customOptions: {}
  }
</script>
<script setup>
    import HelloWorld from '../components/HelloWorld.vue'
    // 在 setup() 作用域中执行 (对每个实例皆如此)
    
</script>
<template>
  <div>
    <HelloWorld msg="Vue3 + TypeScript + Vite"/>
  </div>
</template>
```

注意：Vue 3 SFC 一般会自动从组件的文件名推断出组件的 name。在大多数情况下，不需要明确的 name 声明。唯一需要的情况是当你需要 `<keep-alive>` 包含或排除或直接检查组件的选项时，你需要这个名字。