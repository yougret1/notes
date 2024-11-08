# 基本认识

OpenGL本身是一种规范，实际上并没有确定任何代码，只是代码规范，他没有任何的库。

不需要下载，显卡生产商来书写，GPU会包含OpenGL的实现，而且所有的显卡生产商会有不同的实现，因此，每个产商对OpenGL的实现都会有细微的不同。产商不可能把GPU的代码开源，因此无法看到源码。

OpenGL跨平台，只需要写一份代码，可以在所有平台运行

顶点着色器转换坐标

片段着色器会为每个像素进行一次去光栅化，决定颜色

## API

docs.gl

#### 报错观看

Output 双击信息，可以看到具体报错信息

##### 初始化

查找办法，二分查找，

```c++
static void GLClearError() {
    while (glGetError() != GL_NO_ERROR) {
    }
}
static void GLCheckError() {
    while (GLenum error = glGetError()) {
        std::cout << "[OpenGL Error] ( " << error << " )" << std::endl;
    }
}
```

debug后，鼠标移动到error，改下显示格式为16进制， Hexadecimal Display

后进入glew.h搜索，就可以知道哪里有问题

### OpenGL debug

`opengl 4.3 -> gl debug message callback`

### Uniform

# Visual Studio

相对路径 

- Debugging -> Working Directory

## Visual studio快捷键

点击 后 shift+alt+点击   范围性选择

ctrl+shift+/ 注释

alt+F12看源码

ctrl+H find all replace

# 项目搭建

## 依赖添加

三处地方

1. c/c++项目附加包含目录(Additional Include Directories)如：
   - $(SolutionDir)/Dependencies\GLFW\include
2. Linker General Additional Library Directories
   - $(SolutionDir)Dependencies\GLFW\lib-vc2022
3. Linker input Additional Dependencies
   - .......... glfw3.lib

#### 扩展

OpenGL特殊扩展glew，glad

#### 静态库和动态库的区别

静态链接库需要单独定义一个宏，不然连接器就会默认是动态

有s的是动态库

一般而言动态库能少很多问题

静态库lib的运行效率一般高于dll调用

##### 静态库额外定义

c/c++ > preprocessor >Preprocessor Definitions 

新增，GLEW_STATIC

or

也可以直接在代码最上面写 `#define GLEW_STATIC`

#### 引入

- `#include<GL/glew.h>`
  - 路径如下：D:\OpenGL_easy\Pro1\OpenGL\Dependencies\glew-2.1.0\include\GL
    include已经被引入到`Additional Include Directories`中，include脚下的GL文件夹



## 项目报错检测

如：

```
Build started at 23:01...
1>------ Build started: Project: OpenGL, Configuration: Debug Win32 ------
1>LINK : warning LNK4098: defaultlib 'MSVCRT' conflicts with use of other libs; use /NODEFAULTLIB:library
1>Application.obj : error LNK2019: unresolved external symbol __imp__glClear@4 referenced in function _main
1>D:\OpenGL_easy\Pro1\OpenGL\Debug\OpenGL.exe : fatal error LNK1120: 1 unresolved externals
1>Done building project "OpenGL.vcxproj" -- FAILED.
========== Build: 0 succeeded, 1 failed, 0 up-to-date, 0 skipped ==========
========== Build completed at 23:01 and took 00.459 seconds ==========
```

仔细看`__imp__glClear`,然后获取glClear去搜索，Microsoft Learn to find `Requirements>Library` 将会看到需要的文件，然后去依赖添加

# 简单测试

## 小三角形

### 简单创建

```c++
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        // 画一个三角形
        glBegin(GL_TRIANGLES);
        glVertex2d(-0.5f, -0.5f);
        glVertex2d(0.0f, 0.5f);
        glVertex2d(0.5f, -0.5f);
        glEnd();

        /* Swap front and back buffers */
        glfwSwapBuffers(window);
```

### 默认着色器创建渲染

使用顶点数组，当不适用着色器的时候GPU会提供默认着色器，也就是说，会完全依赖驱动。

```c++
    float positions[6] = {
        -0.5f, -0.5f,
        0.0f,   0.5f,
        0.5f,  -0.5f
    };

    unsigned int buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(float), positions, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0); //启用或禁用通用顶点属性数组
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);//定义通用顶点数据
```

 

### 使用shader来写三角形

```c++
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

static unsigned int CompileShader(unsigned int type,const std::string& source ) {
    unsigned int id = glCreateShader(type);//创建新的顶点着色器对象
    const char* src = source.c_str();//把source变量转换为C风格的字符串数组
    glShaderSource(id, 1, &src, nullptr);//把着色器的源码设置到着色器对象上，id:着色器对象的id，1:传输源码字符串数组的数量，src指向源码字符串的指针数组，nullptr，表示字符串以空字符结尾
    glCompileShader(id);//编译指定的着色器对象

    // TODO: ERROR handling
    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE) {
        //获取编译错误日志
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char* message = (char*)alloca(length * sizeof(char));
        glGetShaderInfoLog(id, length, &length, message);
        //打印错误消息
        std::cout << "Failed to compile" << 
            (type == GL_VERTEX_SHADER ? "vertex" : "fragment") 
            << "shader!" << std::endl;
        std::cout << message << std::endl;
        //删除着色器对象
        glDeleteShader(id);
        return 0;


    }

    return id;

}

//using namespace std;
static int CreateShader(const std::string& vertexShader, const std::string& fragmentShader) {
    unsigned int program = glCreateProgram();//创建新的程序对象，只有正整数
    unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
    unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);

    glAttachShader(program, vs);//把编译好的顶点着色器对象vs附加到着色器程序对象上
    glAttachShader(program, fs);
    glLinkProgram(program);//连接着色器程序对象
    glValidateProgram(program);//验证着色器是否有效，检查是否能正确运行

    glDeleteShader(vs);//删除已经编译的顶点着色器对象vs
    glDeleteShader(fs);

    return program;
}

int main(void)
{
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;//初始化glfw库

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current 设置上下文为创建的窗口 */
    glfwMakeContextCurrent(window);


    if (glewInit() != GLEW_OK)
        std::cout << "Error" << std::endl;//初始化glew

    std::cout << glGetString(GL_VERSION) << std::endl;//打印OpenGL版本

    float positions[6] = {
        -0.5f, -0.5f,
        0.0f,   0.5f,
        0.5f,  -0.5f
    };

    unsigned int buffer;
    glGenBuffers(1, &buffer);//1表示只创建一个缓冲区对象，&buffer用于创建新创建的缓冲区对象的id
    glBindBuffer(GL_ARRAY_BUFFER, buffer);//把创建的缓冲区对象绑定到GL_ARRAY_BUFFER目标上，后续的缓冲区操作都会作用域这个绑定的缓冲区对象
    glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(float), positions, GL_STATIC_DRAW);//把顶点数据positions复制到当前绑定的缓冲区区对象中，第二位是数据的总字节大小，GL_STATIC_DRAW表示数据不会被频繁修改，OpenGl可以对其进行优化

    glEnableVertexAttribArray(0); //启用顶点属性数组0，告诉OpenGL如何解释顶点数据
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);//调用顶点属性0的格式，每个顶点有2个浮点值，GL_FLOAT为浮点数，GL_FALSE表示数据不需要被归一化，第四位表示顶点数据的字节跨度，0表示数据在缓冲区中的起使偏移量

    std::string vertexShader =
        "#version 330 core\n"
        "\n"
        "layout(location = 0)in vec4 position;\n"
        "\n"
        "void main(){\n"
        "\n"
        "   gl_Position = position;\n"
        "}\n"
        ;

    std::string fragmentShader =
        "#version 330 core\n"
        ";\n"
        "layout(location = 0) out vec4 color;\n"
        ";\n"
        "void main()\n"
        "{\n"
        "   color = vec4(1.0,0.0,0.0,1.0);\n"
        "}\n"
        ;

    unsigned int shader = CreateShader(vertexShader, fragmentShader);
    glUseProgram(shader);//使用着色器程序

    
    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);//清除颜色缓冲区，确保画布上没有遗留的图形

        glDrawArrays(GL_TRIANGLES,0,3);//表示从第0个元素开始绘制，表示绘制三个顶点

        /* Swap front and back buffers */
        glfwSwapBuffers(window);//这句代码调用 glfwSwapBuffers 函数交换前后缓冲区。OpenGL使用双缓冲技术, 前缓冲区显示在屏幕上, 后缓冲区存储绘制的内容。

        /* Poll for and process events */
        glfwPollEvents();//处理窗口事件，检查输入操作，把事件传递给应用程序处理
    }

    glDeleteProgram(shader);

    glfwTerminate();//终止GLFW
    return 0;
}
```

### 

## uniform引入

```c++
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#define ASSERT(x) if (!(x)) __debugbreak();
#define GLCall(x) GLClearError();\
    x;\
    ASSERT(GLLogCall(#x, __FILE__ , __LINE__))

static void GLClearError() {
	while (glGetError() != GL_NO_ERROR) {
	}
}
static void GLCheckError() {
	while (GLenum error = glGetError()) {
		std::cout << "[OpenGL Error] ( " << error << " )" << std::endl;
	}
}

static bool GLLogCall(const char* function, const char* file, int line) {
	while (GLenum error = glGetError()) {
		std::cout << "[OpenGL Error] (" << error << ")" << function << " " << file << ":" << line << std::endl;
		return false;
	}
	return true;
}

struct ShaderProgramSource {
	std::string VertexSource;
	std::string FragmentSource;
};

static ShaderProgramSource ParseShader(const std::string& filepath) {
	std::ifstream stream(filepath);

	enum class ShaderType {
		NONE = -1, VERTEX = 0, FRAGMENT = 1,
	};

	std::string line;
	std::stringstream ss[2];
	ShaderType type = ShaderType::NONE;
	while (getline(stream, line)) {
		if (line.find("#shader") != std::string::npos) {
			if (line.find("vertex") != std::string::npos) {
				// set mode to vertex
				type = ShaderType::VERTEX;
			}
			else if (line.find("fragment") != std::string::npos) {
				// set mode to fragment
				type = ShaderType::FRAGMENT;
			}
		}
		else {
			ss[(int)type] << line << '\n';
		}
	}
	return { ss[0].str(), ss[1].str() };
}

static unsigned int CompileShader(unsigned int type, const std::string& source) {
	unsigned int id = glCreateShader(type);//创建新的顶点着色器对象
	const char* src = source.c_str();//把source变量转换为C风格的字符串数组
	glShaderSource(id, 1, &src, nullptr);//把着色器的源码设置到着色器对象上，id:着色器对象的id，1:传输源码字符串数组的数量，src指向源码字符串的指针数组，nullptr，表示字符串以空字符结尾
	glCompileShader(id);//编译指定的着色器对象

	// TODO: ERROR handling
	int result;
	glGetShaderiv(id, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE) {
		//获取编译错误日志
		int length;
		glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
		char* message = (char*)alloca(length * sizeof(char));
		glGetShaderInfoLog(id, length, &length, message);
		//打印错误消息
		std::cout << "Failed to compile" <<
			(type == GL_VERTEX_SHADER ? "vertex" : "fragment")
			<< "shader!" << std::endl;
		std::cout << message << std::endl;
		//删除着色器对象
		glDeleteShader(id);
		return 0;
	}
	return id;
}

//using namespace std;
static int CreateShader(const std::string& vertexShader, const std::string& fragmentShader) {
	unsigned int program = glCreateProgram();//创建新的程序对象，只有正整数
	unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
	unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);

	glAttachShader(program, vs);//把编译好的顶点着色器对象vs附加到着色器程序对象上
	glAttachShader(program, fs);
	glLinkProgram(program);//连接着色器程序对象
	glValidateProgram(program);//验证着色器是否有效，检查是否能正确运行

	glDeleteShader(vs);//删除已经编译的顶点着色器对象vs
	glDeleteShader(fs);

	return program;
}

int main(void)
{
	GLFWwindow* window;

	/* Initialize the library */
	if (!glfwInit())
		return -1;//初始化glfw库

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	/* Make the window's context current 设置上下文为创建的窗口 */
	glfwMakeContextCurrent(window);

	glfwSwapInterval(1);//开启垂直同步

	if (glewInit() != GLEW_OK)
		std::cout << "Error" << std::endl;//初始化glew

	std::cout << glGetString(GL_VERSION) << std::endl;//打印OpenGL版本

	float positions[] = {
		-0.5f, -0.5f,
		0.5f,  -0.5f,
		0.5f,   0.5f,
		-0.5f,  0.5f
	};
	unsigned int indices[]{
		0 , 1 , 2,
		2 , 3 , 1,
	};

	unsigned int buffer;
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(float), positions, GL_STATIC_DRAW);//把数据绑定到GL_ARRAY_BUFFER当中
	//glBindBuffer只是将缓冲区对象关联到特定的目标上, 而glBufferData实际上将数据传输到了GPU内存中的缓冲区对象中。

	glEnableVertexAttribArray(0); //启用顶点属性数组0，告诉OpenGL如何解释顶点数据
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);//调用顶点属性0的格式，每个顶点有2个浮点值，GL_FLOAT为浮点数，GL_FALSE表示数据不需要被归一化，第四位表示顶点数据的字节跨度，0表示数据在缓冲区中的起使偏移量

	unsigned int ibo;
	GLCall(glGenBuffers(1, &ibo));//1表示只创建一个缓冲区对象，&buffer用于创建新创建的缓冲区对象的id
	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo));//把创建的缓冲区对象绑定到GL_ELEMENT_ARRAY_BUFFER目标上，后续的缓冲区操作都会作用域这个绑定的缓冲区对象
	GLCall(glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(unsigned int), indices, GL_STATIC_DRAW));//把顶点数据positions复制到当前绑定的缓冲区区对象中，第二位是数据的总字节大小，GL_STATIC_DRAW表示数据不会被频繁修改，OpenGl可以对其进行优化

	ShaderProgramSource source = ParseShader("res/shaders/Basic.shader");//

	//std::cout << "VERTEX\n" << source.VertexSource << std::endl;
	//std::cout << "FRAGMENT\n" << source.FragmentSource << std::endl;

	unsigned int shader = CreateShader(source.VertexSource, source.FragmentSource);
	GLCall(glUseProgram(shader));//使用着色器程序

	int location = glGetUniformLocation(shader, "u_Color");//如果找不到uniform，那么就返回-1，如果没有使用shader，但是定义了shader，那么opengl会在使用的时候gc掉shader
	//int location = -1;
	ASSERT(location != -1);
	GLCall(glUniform4f(location, 0.2f, 0.3f, 0.8f, 1.0f));
	float r = 0.0f;
	float increment = 0.01f;

	/* Loop until the user closes the window */
	while (!glfwWindowShouldClose(window))
	{
		/* Render here */
		glClear(GL_COLOR_BUFFER_BIT);//清除颜色缓冲区，确保画布上没有遗留的图形

		GLCall(glUniform4f(location, r, 0.3f, 0.8f, 1.0f));

		GLCall(glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr));

		if (r > 1.0f)
			increment = -0.001f;
		else if (r < 0.0f)
			increment = 0.001f;
		r += increment;

		/* Swap front and back buffers */
		glfwSwapBuffers(window);//这句代码调用 glfwSwapBuffers 函数s交换前后缓冲区。OpenGL使用双缓冲技术, 前缓冲区显示在屏幕上, 后缓冲区存储绘制的内容。

		/* Poll for and process events */
		glfwPollEvents();//处理窗口事件，检查输入操作，把事件传递给应用程序处理
	}

	glDeleteProgram(shader);

	glfwTerminate();//终止GLFW
	return 0;
}
```

