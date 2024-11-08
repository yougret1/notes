# 虚拟环境

## 创建虚拟环境

conda create -n py311 python=3.11

## 激活虚拟环境

conda activate py311

## 检查python版本

python

## 目录选择

cd E:/......

## 查看当前环境

conda info --envs

# 其他

利用pip将当前Python环境中已安装的所有包的包名和版本信息，输出重定向到指定文件中

> *#输出已经安装的所有包的包名和版本信息* 
>
> pip freeze 
>
> #将上述结果输出重定向到requirements.txt文件中保存
>
> pip freeze > requirements.txt

[2] 批量安装requirements.txt中指定的包

```bash
pip install -r requirements.txt
```

## [conda]

[1] 批量导出

```bash
conda list -e > requirements.txt
```

[2] 批量安装

```bash
conda install --yes --file requirements.txt
```