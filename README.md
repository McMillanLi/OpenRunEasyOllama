# Open Run Easy Ollama (OREO)

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

基于Ollama的本地大模型客户端，提供直观的图形界面管理AI模型并进行对话交互。

## 功能特性

### 🚀 核心功能
- **自然对话交互**  
  支持流式响应、对话历史管理和Markdown渲染
- **全功能模型管理**  
  模型下载/删除/刷新，实时进度显示
- **智能参数调节**  
  温度值实时调节（0.1~2.0范围）
- **跨平台支持**  
  兼容Windows/macOS/Linux系统

### 🛠️ 技术亮点
- 线程安全的网络请求
- 响应式UI设计（1024x768自适应布局）
- 完善的中文错误提示系统
- 基于requests库的高效通信

## 安装与运行

### 环境要求
- Python 3.8+
- Ollama服务（本地运行在11434端口）

### 快速开始
```bash
# 克隆仓库
git clone https://github.com/McMillanLi/OpenRunEasyOllama.git


# 安装依赖
pip install -r requirements.txt

# 启动程序
python OREO.py