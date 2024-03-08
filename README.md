# sd-webui-next-style

这是一个适用于SD WebUI的多样绘画风格选择器，现有近千种风格可供选择！

这个插件基于下面这个项目：

>https://github.com/javsezlol1/Stylez

我所进行的修改：
- [x] 添加Fooocus全部风格模板，删除预设风格中的重复项。
- [x] 对插件界面进行完全汉化。
- [x] 改进风格卡片选择的交互性，对已选择风格进行标记。
- [ ] 改进风格卡片选择的交互性，将已选择风格卡片进行顶置。
- [ ] 加入工作流框架。
<img src="https://bu.dusays.com/2024/03/09/65eb37a3ae677.png" alt="Alt text" title="Optional title">

# 安装 

### 插件安装

>正常启动webui（确保您的webui是最新的）点开扩展>从网址安装>并将 https://github.com/Firetheft/sd-webui-next-style.git 粘贴到扩展的git仓库网址中，然后点击安装。最后记得应用更改并重启！
<img src="https://bu.dusays.com/2024/03/08/65eb0fa98a046.gif" alt="Alt text" title="Optional title">

### 风格生成器模型下载

>打开extension/sd-webui-next-style文件夹，右键打开power shell

>首先输入：git lfs installgit

>然后输入：git clone https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2

# 使用

sd-webui-next-style安装在extensions文件夹中，它使用一些CSS和JS来删除旧的风格。但是不要担心，在第一次启动时，所有保存的风格都会转换为单独的JSON文件。这将允许您将它们与库一起使用。添加空白预览图像，以便以后根据需要进行更改。

<img src="https://bu.dusays.com/2024/03/08/65eb0f527b1e0.gif" alt="Alt text" title="Optional title">

