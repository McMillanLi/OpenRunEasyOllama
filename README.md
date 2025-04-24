# Open Run Easy Ollama (OREO)

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

一个现代化、用户友好的Ollama客户端，提供直观的图形界面管理AI模型并进行对话交互。

## 功能特性

### 🚀 核心功能
- **自然对话交互**  
  支持流式响应、完整对话历史管理和思维链过滤
- **全功能模型管理**  
  模型下载/删除/刷新，实时进度显示，支持浏览Ollama官方模型库
- **智能参数调节**  
  温度值实时调节（0.1~2.0范围），影响模型创造性输出
- **跨平台支持**  
  兼容Windows/macOS/Linux系统，统一的用户体验

### 🛠️ 技术亮点
- 美观现代的界面设计，配色协调
- 简化Ollama的使用流程，适合新手用户
- 智能流式输出处理，避免内容重复
- 支持思维链过滤，只展示最终回答
- 线程安全的网络请求和异步UI更新
- 响应式UI设计（1024x768自适应布局）
- 完善的中文错误提示和状态反馈
- 基于requests库的高效通信，支持超时检测
- 人性化的错误处理机制

## 安装与运行

### 环境要求
- Python 3.8+
- Ollama服务（本地运行在11434端口）

### 快速开始
```bash
# 克隆仓库
git clone https://github.com/McMillanLi/OpenRunEasyOllama.git
```
```bash
# 安装依赖
pip install -r requirements.txt
```
```bash
# 启动程序
python OREO.py
```

## 使用指南

### 主界面

![主界面](path/to/main_ui.png)

主界面由以下部分组成：
- **顶部控制栏**：包含模型选择下拉框、温度调节滑块和模型管理按钮
- **对话历史区**：显示用户与AI的对话内容，支持流式输出
- **输入区域**：用于输入对话内容，支持按Enter键发送
- **状态栏**：显示当前程序状态和操作反馈

### 模型管理

![模型管理](path/to/model_management.png)

模型管理界面提供以下功能：
- **模型下载**：输入模型名称，点击"下载模型"按钮下载
- **浏览模型**：点击"浏览模型"按钮打开Ollama官方模型库
- **模型列表**：显示已安装的模型，包含名称、哈希值、大小和更新时间
- **操作按钮**：刷新列表、删除选中模型和返回聊天界面

### 自定义配置

如需修改默认连接设置，可编辑OREO.py文件中的以下部分：

```python
class OllamaClient:
    def __init__(self,base_url: str = "http://192.168.110.38:11451",
            model: str = "deepseek-r1:14b",
            timeout: int = 300,
            system_prompt: Optional[str] = None
    ):
```

将base_url修改为您的Ollama服务地址和端口。

## 常见问题

1. **连接错误**
   - 确保Ollama服务已启动
   - 检查配置文件中的IP地址和端口是否正确
   - 检查网络连接是否正常

2. **模型下载失败**
   - 确保网络连接稳定
   - 确认模型名称正确（可从Ollama模型库查看）
   - 检查存储空间是否充足

3. **思维链输出重复**
   - 最新版本已修复此问题，请确保使用最新版本

## 贡献指南

欢迎贡献代码、报告问题或提出功能建议。请遵循以下步骤：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 许可证

本项目基于MIT许可证开源 - 详情请参阅 [LICENSE](LICENSE) 文件

## 致谢

- [Ollama](https://ollama.com) - 提供本地大语言模型运行环境
- [Tkinter](https://docs.python.org/3/library/tkinter.html) - Python标准GUI库
- 所有贡献者和用户