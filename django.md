## 连接mysql

### 安装`PyMySQL`包

### 在项目同名包下添加

```python
import pymysql
pymysql.install_as_MySQLdb()
```

### setting.py数据库设置

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'your_database_name',
        'USER': 'your_mysql_username',
        'PASSWORD': 'your_mysql_password',
        'HOST': 'localhost',
        'PORT': '3306',
    }
}

```

### 数据迁移

`python manage.py migrate`

## 创建超级管理员账号(必要)

`python manage.py createsuperuser`