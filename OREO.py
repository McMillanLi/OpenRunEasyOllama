import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from threading import Thread
import tkinter.font
import requests
import json
import webbrowser
from typing import Generator, Optional, Dict, List


class OllamaClient:
    def __init__(self,base_url: str = "http://192.168.110.38:11451",
            model: str = "deepseek-r1:14b",
            timeout: int = 300,
            system_prompt: Optional[str] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.messages: List[Dict[str, str]] = []

        if system_prompt and system_prompt.strip():
            self.messages.append({
                "role": "system",
                "content": system_prompt.strip()
            })

    def chat(
            self,
            prompt: str,
            stream: bool = False,
            temperature: float = 0.6,
            max_tokens: int = 1000,
            **kwargs
    ) -> Generator[str, None, None] | Dict:
        # 添加用户消息到历史
        self.messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs
            }
        }

        try:
            # 使用较小的请求超时，这样如果服务未响应可以快速失败
            initial_timeout = 5  # 初始连接超时5秒
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=(initial_timeout, self.timeout),  # (连接超时, 读取超时)
                stream=stream
            )
            response.raise_for_status()
        except requests.exceptions.ConnectTimeout:
            self.messages.pop()  # 移除最后添加的用户消息
            raise RuntimeError(f"连接超时，无法连接到 {self.base_url}，请检查Ollama服务是否已启动") 
        except requests.exceptions.ReadTimeout:
            self.messages.pop()  # 移除最后添加的用户消息
            raise RuntimeError(f"请求超时，模型响应时间过长")
        except requests.exceptions.ConnectionError:
            self.messages.pop()
            raise RuntimeError(f"连接错误，无法连接到 {self.base_url}，请检查网络和Ollama服务状态")
        except requests.exceptions.RequestException as e:
            self.messages.pop()  # 移除最后添加的用户消息
            raise RuntimeError(f"API 请求失败: {str(e)}") from e

        if stream:
            return self._handle_stream_response(response)
        else:
            return self._handle_full_response(response)

    def _handle_stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        full_response = ""
        
        try:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    content = chunk.get("message", {}).get("content", "")
                    
                    # 过滤掉思维链标记
                    filtered_content = self._filter_thinking_tags(content)
                    
                    # 计算真正的增量内容 (去除已有内容)
                    if filtered_content:
                        # 如果full_response是filtered_content的前缀，只返回新增部分
                        if filtered_content.startswith(full_response):
                            delta = filtered_content[len(full_response):]
                            if delta:  # 只有有新内容时才yield
                                yield delta
                                full_response = filtered_content
                        else:
                            # 内容不连续，可能是模型重新生成了内容
                            # 这种情况下，我们返回完整的新内容
                            yield filtered_content
                            full_response = filtered_content
            
            # 存储最终的完整响应
            final_response = self._filter_thinking_tags(full_response)
            self._append_assistant_message(final_response)
            
        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"响应解析失败: {str(e)}") from e
            
    def _filter_thinking_tags(self, text):
        """过滤掉思维链标记和内容"""
        # 如果没有思维链标记，直接返回原文本
        if "<think>" not in text:
            return text
            
        # 过滤掉所有<think>...</think>内容
        result = ""
        start_index = 0
        
        while True:
            think_start = text.find("<think>", start_index)
            if think_start == -1:
                # 没有更多的<think>标记，添加剩余文本
                result += text[start_index:]
                break
                
            # 添加<think>之前的内容
            result += text[start_index:think_start]
            
            # 查找配对的</think>
            think_end = text.find("</think>", think_start)
            if think_end == -1:
                # 没有找到闭合标签，剩余部分被认为是思维链，忽略
                break
                
            # 移动到</think>之后继续处理
            start_index = think_end + 8
        
        return result

    def _append_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def _handle_full_response(self, response: requests.Response) -> Dict:
        """
        处理完整响应
        """
        try:
            data = response.json()
            if "message" in data:
                self.messages.append(data["message"])
            return data
        except json.JSONDecodeError as e:
            raise RuntimeError(f"响应解析失败: {str(e)}") from e

    def clear_history(self) -> None:
        """清空对话历史"""
        self.messages = []

    def get_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """获取对话历史"""
        return self.messages[-last_n:] if last_n else self.messages.copy()

    def get_installed_models(self) -> List[Dict]:
        """获取已安装的模型列表"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"获取模型列表失败: {str(e)}") from e

    def download_model(self, name: str, progress_callback: callable):
        def _stream_download():
            try:
                response = requests.post(
                    f"{self.base_url}/api/pull",
                    json={"name": name},
                    stream=True,
                    timeout=self.timeout
                )
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        # 统一处理进度信息
                        progress = {
                            "model": name,# 始终包含模型名称
                            "status": chunk.get("status", ""),
                            "completed": chunk.get("completed", 0),
                            "total": chunk.get("total", 1),
                            "percentage": chunk.get("completed", 0) / chunk.get("total", 1) * 1
                        }
                        progress_callback(progress)

                progress_callback({
                    "model": name,
                    "status": "下载完成",
                    "percentage": 100
                })

            except requests.exceptions.RequestException as e:
                progress_callback({"error": str(e), "model": name})
                raise RuntimeError(f"下载失败: {str(e)}") from e

        # 启动下载线程
        Thread(target=_stream_download, daemon=True).start()

    def delete_model(self, name: str):
        """删除指定模型"""
        try:
            response = requests.delete(
                f"{self.base_url}/api/delete",
                json={"name": name},
                timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"删除失败: {str(e)}") from e


class OllamaGUI:
    def __init__(self, master):
        self.master = master
        master.title("Open Run Easy Ollama")
        master.geometry("1024x768")
        master.configure(bg='#F5F7FA')  # 设置背景颜色

        # 配置全局字体和颜色
        default_font = ('Microsoft YaHei', 11)
        tk_font = tk.font.nametofont("TkDefaultFont")
        tk_font.configure(family=default_font[0], size=default_font[1])
        
        # 初始化客户端
        self.client = OllamaClient()
        self.streaming = False  # 流式响应状态标志
        self.current_response = ""  # 当前响应内容
        self.progress_window = None
        self.progress_bar = None
        self.progress_label = None

        # 添加DPI感知设置
        self.style = ttk.Style()
        self._setup_styles()  # 样式配置
        
        # 创建界面组件
        self.create_widgets()
        self.update_model_menu()
        self.update_status("准备就绪 | 模型: " + self.client.model)

        # 添加文本标签样式
        self.history_area.tag_configure('user', foreground='#0066CC', font=('Microsoft YaHei Bold', 11))
        self.history_area.tag_configure('assistant', foreground='#006633', font=('Microsoft YaHei Bold', 11))
        self.history_area.tag_configure('error', foreground='#CC0000', font=('Microsoft YaHei', 11))
        self.history_area.tag_configure('waiting', foreground='#888888', font=('Microsoft YaHei', 11, 'italic'))

    def _setup_styles(self):
        """配置风格样式"""
        self.style.theme_use('default')
        
        # 全局字体和颜色
        self.style.configure('.',
            font=('Microsoft YaHei', 11),
            foreground='#2C3E50',
            background='#F5F7FA')
            
        # 标签样式
        self.style.configure('TLabel',
            background='#F5F7FA',
            foreground='#2C3E50',
            padding=3)
            
        # 按钮样式
        self.style.configure('TButton',
            borderwidth=0,
            padding=6,
            relief='flat',
            background='#3498DB',
            foreground='#FFFFFF',
            font=('Microsoft YaHei', 11, 'bold'))
        self.style.map('TButton',
            background=[('active', '#2980B9'), ('pressed', '#1A5276')],
            foreground=[('active', '#FFFFFF'), ('pressed', '#ECF0F1')])
            
        # 输入框样式
        self.style.configure('TEntry',
            fieldbackground='#FFFFFF',
            bordercolor='#BDC3C7',
            borderwidth=1,
            relief='solid',
            padding=(8, 5),
            font=('Microsoft YaHei', 11))
            
        # 下拉框样式
        self.style.configure('TCombobox',
            fieldbackground='#FFFFFF',
            background='#3498DB',
            arrowsize=14,
            arrowcolor='#2C3E50',
            padding=(6, 3),
            font=('Microsoft YaHei', 11))
        self.style.map('TCombobox',
            fieldbackground=[('readonly', '#FFFFFF')],
            selectbackground=[('readonly', '#3498DB')],
            selectforeground=[('readonly', '#FFFFFF')])
            
        # 进度条样式
        self.style.configure('Horizontal.TProgressbar',
            thickness=8,
            troughcolor='#ECF0F1',
            background='#3498DB',
            bordercolor='#ECF0F1')
            
        # 滑块样式
        self.style.configure('Horizontal.TScale',
            troughcolor='#D6EAF8',
            sliderthickness=16,
            sliderrelief='flat',
            background='#3498DB')
        self.style.map('Horizontal.TScale',
            background=[('active', '#2980B9'), ('pressed', '#1A5276')],
            troughcolor=[('active', '#AED6F1')])
            
        # 自定义温度滑块样式
        self.style.configure('Temperature.Horizontal.TScale',
            troughcolor='#D6EAF8',
            sliderthickness=18,
            sliderrelief='flat',
            background='#3498DB')
        self.style.map('Temperature.Horizontal.TScale',
            background=[('active', '#2ECC71'), ('pressed', '#27AE60')],
            troughcolor=[('active', '#BBDEFB'), ('pressed', '#BBDEFB')])
            
        # 表格样式
        self.style.configure('Treeview',
            rowheight=28,
            fieldbackground='#FFFFFF',
            background='#FFFFFF',
            foreground='#2C3E50',
            bordercolor='#E5E8E8',
            font=('Microsoft YaHei', 11))
        self.style.configure('Treeview.Heading',
            font=('Microsoft YaHei', 12, 'bold'),
            background='#D6EAF8',
            foreground='#2C3E50',
            relief='flat')
        self.style.map('Treeview',
            background=[('selected', '#3498DB')],
            foreground=[('selected', '#FFFFFF')])

    def update_model_menu(self):
        """更新下拉菜单的模型列表"""
        try:
            models = self.client.get_installed_models()
            self.model_menu['values'] = [model['name'] for model in models]
        except Exception as e:
            messagebox.showerror("错误", f"加载模型列表失败: {str(e)}")

    def create_widgets(self):
        # 顶部控制栏
        self.control_frame = ttk.Frame(self.master, padding=10, style='TFrame')
        self.control_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        # 模型选择
        ttk.Label(self.control_frame, text="模型:", style='TLabel').pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=self.client.model)
        self.model_menu = ttk.Combobox(self.control_frame, textvariable=self.model_var, width=20)
        self.model_menu.pack(side=tk.LEFT, padx=5)
        
        # 参数设置
        ttk.Label(self.control_frame, text="温度:", style='TLabel').pack(side=tk.LEFT, padx=(15, 0))
        self.temp_value = ttk.Label(self.control_frame, text="0.6", width=3, style='TLabel')
        self.temp_value.pack(side=tk.LEFT)
        self.temp_slider = ttk.Scale(self.control_frame, from_=0.1, to=2.0, value=0.6,
                                   command=lambda v: self.temp_value.config(text=f"{float(v):.1f}"),
                                   style='Temperature.Horizontal.TScale')
        self.temp_slider.pack(side=tk.LEFT, padx=5)

        # 在控制栏最右侧添加模型管理按钮
        ttk.Button(self.control_frame, text="模型管理", command=self.show_model_manager).pack(side=tk.RIGHT, padx=5)

        # 对话历史显示
        history_frame = ttk.Frame(self.master, padding=5)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.history_area = scrolledtext.ScrolledText(
            history_frame,
            wrap=tk.WORD,
            state='disabled',
            font=('Microsoft YaHei', 11),
            background='#FFFFFF',
            foreground='#2C3E50',
            padx=10,
            pady=10,
            borderwidth=1,
            relief=tk.SOLID
        )
        self.history_area.pack(fill=tk.BOTH, expand=True)

        # 输入区域
        self.input_frame = ttk.Frame(self.master, padding=5)
        self.input_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        self.input_entry = ttk.Entry(self.input_frame, font=('Microsoft YaHei', 11))
        self.input_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        self.input_entry.bind("<Return>", self.send_message)

        ttk.Button(
            self.input_frame,
            text="发送",
            command=self.send_message,
            width=6,
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            self.input_frame,
            text="清空",
            command=self.clear_history,
            width=6,
        ).pack(side=tk.LEFT, padx=2)

        # 状态栏
        status_frame = ttk.Frame(self.master, padding=(5, 2))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_bar = ttk.Label(
            status_frame,
            relief=tk.FLAT,
            anchor=tk.W,
            padding=(5, 2),
            background='#EBF5FB'
        )
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 文本样式配置
        self.history_area.tag_configure('user', foreground='#2980B9', font=('Microsoft YaHei Bold', 11))
        self.history_area.tag_configure('assistant', foreground='#16A085', font=('Microsoft YaHei Bold', 11))
        self.history_area.tag_configure('error', foreground='#E74C3C', font=('Microsoft YaHei', 11))
        self.history_area.tag_configure('waiting', foreground='#7F8C8D', font=('Microsoft YaHei', 11, 'italic'))

    def show_model_manager(self):
        """显示模型管理界面"""
        # 隐藏主界面组件
        self.control_frame.pack_forget()
        self.history_area.master.pack_forget()
        self.input_frame.pack_forget()
        
        # 清除可能存在的其他组件
        for widget in self.master.winfo_children():
            if widget != self.status_bar.master:  # 保留状态栏
                widget.pack_forget()

        # 创建模型管理面板
        self.model_manager_frame = ttk.Frame(self.master)
        self.model_manager_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 管理界面标题
        header_frame = ttk.Frame(self.model_manager_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            header_frame, 
            text="模型管理", 
            font=('Microsoft YaHei', 16, 'bold'),
            foreground='#2C3E50'
        ).pack(side=tk.LEFT, pady=0)

        # 添加模型操作区域
        operation_frame = ttk.Frame(self.model_manager_frame)
        operation_frame.pack(fill=tk.X, pady=5)

        # 模型搜索/下载输入框
        ttk.Label(operation_frame, text="模型名称:", width=8).pack(side=tk.LEFT, padx=(0, 5))
        self.new_model_var = tk.StringVar()
        ttk.Entry(operation_frame, textvariable=self.new_model_var, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            operation_frame,
            text="下载模型",
            command=self.download_model
        ).pack(side=tk.LEFT, padx=5)
        
        # 添加浏览模型按钮
        ttk.Button(
            operation_frame,
            text="浏览模型",
            command=self.browse_models
        ).pack(side=tk.LEFT, padx=5)

        # 模型列表
        tree_frame = ttk.Frame(self.model_manager_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 添加滚动条
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.model_tree = ttk.Treeview(
            tree_frame,
            columns=('name', 'hash', 'size', 'modified'),
            show='headings',
            selectmode='browse',
            style='Treeview',
            yscrollcommand=tree_scroll.set
        )
        tree_scroll.config(command=self.model_tree.yview)
        
        # 设置列宽和标题
        self.model_tree.column('name', width=250, stretch=tk.YES)
        self.model_tree.column('hash', width=100, anchor=tk.CENTER)
        self.model_tree.column('size', width=100, anchor=tk.CENTER)
        self.model_tree.column('modified', width=150, anchor=tk.CENTER)
        
        self.model_tree.heading('name', text='模型名称')
        self.model_tree.heading('hash', text='哈希值')
        self.model_tree.heading('size', text='大小')
        self.model_tree.heading('modified', text='更新时间')
        self.model_tree.pack(fill=tk.BOTH, expand=True)

        # 操作按钮
        btn_frame = ttk.Frame(self.model_manager_frame)
        btn_frame.pack(pady=5)
        
        ttk.Button(btn_frame, text="刷新列表", width=10, command=self.refresh_model_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="删除模型", width=10, command=self.delete_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="返回聊天", width=10, command=self.return_to_chat).pack(side=tk.LEFT, padx=5)

        # 初始化模型列表
        self.refresh_model_list()

    def return_to_chat(self):
        """返回聊天主界面"""
        # 清理所有窗口组件
        for widget in self.master.winfo_children():
            if widget != self.status_bar.master:  # 保留状态栏
                widget.destroy()
        
        # 重新创建主界面组件
        self.create_widgets()
        self.update_model_menu()

    def update_status(self, message):
        self.status_bar.config(text=message)

    def append_to_history(self, text, role=None):
        """添加内容到历史记录"""
        self.history_area.config(state='normal')

        if role == 'user':
            self.history_area.insert(tk.END, "用户: ", 'user')
            self.history_area.insert(tk.END, text + "\n\n")
        elif role == 'assistant':
            self.history_area.insert(tk.END, "AI: ", 'assistant')
            self.history_area.insert(tk.END, text + "\n\n")
        elif role == 'error':
            self.history_area.insert(tk.END, "错误: ", 'error')
            self.history_area.insert(tk.END, text + "\n\n")
        
        self.history_area.config(state='disabled')
        self.history_area.see(tk.END)

    def clear_history(self):
        """清空对话历史"""
        self.client.clear_history()
        self.history_area.config(state='normal')
        self.history_area.delete(1.0, tk.END)
        self.history_area.config(state='disabled')
        self.update_status("对话历史已清空")

    def send_message(self, event=None):
        """处理用户发送消息"""
        # 如果正在处理响应，忽略新的请求
        if self.streaming:
            return
            
        # 获取并验证用户输入
        prompt = self.input_entry.get().strip()
        if not prompt:
            return
            
        # 更新模型选择
        try:
            new_model = self.model_var.get()
            self.client.model = new_model
            self.update_status(f"当前模型: {new_model}")
        except Exception as e:
            messagebox.showerror("错误", f"模型切换失败: {str(e)}")
            return

        # 清空输入框
        self.input_entry.delete(0, tk.END)
        
        # 显示用户输入到历史记录
        self.append_to_history(prompt, 'user')
        
        # 启动后台线程处理响应
        Thread(target=self.process_response, args=(prompt,), daemon=True).start()

    def process_response(self, prompt):
        """处理模型响应"""
        self.streaming = True
        self.current_response = ""
        
        # 先显示AI回复的开始标记
        self.history_area.config(state='normal')
        self.history_area.insert(tk.END, "AI: ", 'assistant')
        self.history_area.config(state='disabled')
        self.history_area.see(tk.END)
        
        try:
            # 获取生成参数
            temperature = round(self.temp_slider.get(), 1)
            
            # 更新状态
            self.update_status(f"生成回复中... | 模型: {self.client.model}")
            
            # 从客户端获取流式响应
            response_stream = self.client.chat(
                prompt,
                stream=True,
                temperature=temperature
            )
            
            # 处理流式响应
            for delta in response_stream:
                if delta:
                    # 添加增量内容
                    self.current_response += delta
                    self.update_response_display(delta)
            
            # 完成响应，添加换行
            self.finalize_response()
            
        except Exception as e:
            # 显示错误信息
            self.streaming = False
            self.history_area.config(state='normal')
            # 删除之前添加的"AI: "
            last_pos = self.history_area.index(tk.END+"-1c linestart")
            self.history_area.delete(last_pos, tk.END)
            self.history_area.config(state='disabled')
            
            # 添加错误信息
            self.append_to_history(str(e), 'error')
            self.update_status(f"错误: {str(e)}")
        finally:
            self.streaming = False
            self.update_status(f"回复完成 | 模型: {self.client.model}")

    def update_response_display(self, delta=None):
        """更新响应显示，只添加增量内容"""
        if not self.streaming:
            return
            
        self.history_area.config(state='normal')
        
        # 获取最后一行的开始位置
        last_line_start = self.history_area.index(tk.END+"-1c linestart")
        
        # 检查最后一行是否以"AI: "开头
        last_line = self.history_area.get(last_line_start, tk.END).strip()
        
        if last_line.startswith("AI:"):
            if delta:
                # 添加增量内容（不是替换全部内容）
                self.history_area.insert(tk.END, delta)
            else:
                # 如果delta为None，完全替换内容
                ai_prefix_end = self.history_area.search(":", last_line_start, tk.END) + "+1c"
                self.history_area.delete(ai_prefix_end, tk.END)
                self.history_area.insert(tk.END, " " + self.current_response)
        else:
            # 这种情况不应该发生，但以防万一
            if delta:
                self.history_area.insert(tk.END, delta)
            else:
                self.history_area.insert(tk.END, self.current_response)
        
        self.history_area.see(tk.END)
        self.history_area.config(state='disabled')
        
        # 强制更新UI
        self.master.update_idletasks()

    def finalize_response(self):
        """完成响应"""
        if not self.streaming:
            return
            
        self.history_area.config(state='normal')
        # 添加额外的换行符
        self.history_area.insert(tk.END, "\n\n")
        self.history_area.config(state='disabled')
        self.history_area.see(tk.END)

    def show_error(self, message):
        """显示错误信息"""
        self.append_to_history(message, 'error')
        self.update_status(f"错误: {message}")

    def refresh_model_list(self):
        """刷新模型列表"""
        self.model_tree.delete(*self.model_tree.get_children())
        try:
            models = self.client.get_installed_models()
            for model in models:
                self.model_tree.insert(
                    '', 'end',
                    values=(
                        model.get('name', '未知模型'),
                        model.get('digest', '')[:12] if model.get('digest') else '',
                        f"{model.get('size', 0)/1024/1024:.1f} MB",
                        model.get('modified_at', '')[:16] if model.get('modified_at') else ''
                    )
                )
            self.update_model_menu()
        except Exception as e:
            messagebox.showerror("错误", f"获取模型列表失败: {str(e)}")

    def download_model(self):
        model_name = self.new_model_var.get().strip()
        if not model_name:
            messagebox.showwarning("提示", "请输入模型名称")
            return


        self.progress_window = tk.Toplevel(self.master)
        self.progress_window.title("下载进度")
        ttk.Label(self.progress_window, text=f"正在下载: {model_name}").pack(pady=5)
        self.progress_bar = ttk.Progressbar(self.progress_window, length=300, mode='determinate')
        self.progress_bar.pack(padx=20, pady=10)
        self.progress_label = ttk.Label(self.progress_window, text="0%")
        self.progress_label.pack(pady=5)

        def download_thread():
            try:
                self.client.download_model(
                    model_name,
                    progress_callback=self.update_download_progress
                )
            except Exception as e:
                self.master.after(10, messagebox.showerror, "错误", f"下载失败: {str(e)}")
                self.master.after(10, self.progress_window.destroy)  # 出错时关闭窗口

        Thread(target=download_thread).start()

    def update_download_progress(self, progress):
        """更新下载进度"""
        self.master.after(10, self._update_progress_ui, progress)

    def _update_progress_ui(self, progress):
        """更新进度显示"""
        if not self.progress_window.winfo_exists():
            return
        if 'error' in progress:
            messagebox.showerror("错误", f"下载失败: {progress['error']}")
            self.progress_window.destroy()
            return

        completed = progress.get("completed", 0)
        total = progress.get('total', 1)  # 避免除零错误
        percentage = completed / total * 100 if total else 0

        # 更新进度条（确保数值合法）
        self.progress_bar['value'] = min(percentage, 100)
        # 更新标签（转换为MB显示）
        mb_completed = completed / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        self.progress_label.config(text=f"{percentage:.1f}% ({mb_completed:.1f}/{mb_total:.1f} MB)")
        # 更新状态栏
        self.update_status(f"下载中: {progress.get('model', '未知模型')} - {progress.get('status', '')}")

        # 处理完成状态
        if percentage >= 100:
            self.update_status(f"下载完成: {progress.get('model', '未知模型')}")
            # 延迟关闭窗口确保界面更新
            self.refresh_model_list()
            self.progress_window.after(1500, self.progress_window.destroy)


    def delete_model(self):
        """删除选中模型"""
        selected = self.model_tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请先选择要删除的模型")
            return

        # 获取选中项的模型名称（现在是在values中的第一个元素）
        values = self.model_tree.item(selected[0])['values']
        if not values or len(values) < 1:
            messagebox.showinfo("错误", "无法获取模型信息")
            return
            
        model_name = values[0]  # 第一列是模型名称
        
        if messagebox.askyesno("确认", f"确定删除模型 {model_name} 吗？"):
            try:
                self.client.delete_model(model_name)
                messagebox.showinfo("成功", f"模型 {model_name} 已成功删除")
                self.refresh_model_list()
            except Exception as e:
                messagebox.showerror("错误", f"删除失败: {str(e)}")

    def browse_models(self):
        """打开Ollama官网的模型库"""
        try:
            webbrowser.open("https://ollama.com/library")
            self.update_status("已打开Ollama模型库网页")
        except Exception as e:
            messagebox.showerror("错误", f"无法打开网页: {str(e)}")




if __name__ == "__main__":
    root = tk.Tk()
    app = OllamaGUI(root)
    root.mainloop()