import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from threading import Thread
import requests
import json
from typing import Generator, Optional, Dict, List


class OllamaClient:
    def __init__(self,base_url: str = "http://localhost:11434",
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
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()
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
                    full_response += content
                    yield content
            self._append_assistant_message(full_response)
        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"响应解析失败: {str(e)}") from e

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

        # 初始化客户端
        self.client = OllamaClient()
        self.streaming = False  # 流式响应状态标志
        self.current_response = ""  # 当前响应内容
        self.progress_window = None
        self.progress_bar = None
        self.progress_label = None

        # 创建界面组件
        self.create_widgets()
        self.update_model_menu()
        self.update_status("准备就绪 | 模型: " + self.client.model)

        # 添加DPI感知设置
        self.style = ttk.Style()
        self._setup_styles()  # 新增样式配置方法

    def _setup_styles(self):
        """配置风格样式"""
        self.style.theme_use('default')
        self.style.configure('.',font=('Microsoft YaHei', 11),foreground='#000000',background='#f0f0f0')
        self.style.configure('TLabel',background='#f0f0f0',foreground='#000000',padding=3)
        self.style.configure('TButton',borderwidth=0,padding=6,relief='flat',background='#80b9ee',foreground='#092642')
        self.style.map('TButton',background=[('active', '#559ce4'), ('pressed', '#0078d4')],foreground=[('active', '#092642'), ('pressed', '#092642')])
        self.style.configure('TEntry',fieldbackground='#d6ebff',bordercolor='#8A8886',borderwidth=1,relief='solid',padding=(6, 3))
        self.style.configure('TCombobox',arrowsize=14,arrowcolor='#000000')
        self.style.configure('Horizontal.TProgressbar',thickness=8,troughcolor='#EDEBE9',background='#0078D4',bordercolor='#EDEBE9')
        self.style.configure('Horizontal.TScale',troughcolor='#d6ebff',sliderthickness=14)
        self.style.configure('Treeview',rowheight=25,fieldbackground='#FFFFFF',bordercolor='#EDEBE9')
        self.style.configure('Treeview.Heading',font=('Microsoft YaHei', 12),background='#add8ff',relief='flat')

    def update_model_menu(self):
        """更新下拉菜单的模型列表"""
        try:
            models = self.client.get_installed_models()
            self.model_menu['values'] = [model['name'] for model in models]
        except Exception as e:
            messagebox.showerror("错误", f"加载模型列表失败: {str(e)}")

    def create_widgets(self):
        # 顶部控制栏
        self.control_frame = ttk.Frame(self.master)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        # 模型选择
        ttk.Label(self.control_frame, text="模型:", style='TLabel').pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=self.client.model)
        self.model_menu = ttk.Combobox(self.control_frame,textvariable=self.model_var,width=20,)
        self.model_menu.pack(side=tk.LEFT, padx=5)
        # 参数设置
        ttk.Label(self.control_frame, text="温度:", style='TLabel').pack(side=tk.LEFT, padx=(10, 0))
        self.temp_value = ttk.Label(self.control_frame, text="0.6", style='TLabel')
        self.temp_value.pack(side=tk.LEFT)
        self.temp_slider = ttk.Scale(self.control_frame, from_=0.1, to=2.0, value=0.6,command=lambda v: self.temp_value.config(text=f"{float(v):.1f}"))
        self.temp_slider.pack(side=tk.LEFT, padx=5)

        # 在控制栏最右侧添加模型管理按钮
        ttk.Button(self.control_frame,text="模型管理",command=self.show_model_manager).pack(side=tk.RIGHT, padx=5)

        # 对话历史显示
        self.history_area = scrolledtext.ScrolledText(self.master,wrap=tk.WORD,state='disabled',font=('Microsoft YaHei', 11))
        self.history_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 输入区域
        self.input_frame = ttk.Frame(self.master)
        self.input_frame.pack(fill=tk.X, padx=5, pady=5)

        self.input_entry = ttk.Entry(self.input_frame, font=('Microsoft YaHei', 11))
        self.input_entry.pack(fill=tk.X, expand=True, side=tk.LEFT)
        self.input_entry.bind("<Return>", self.send_message)

        ttk.Button(
            self.input_frame,
            text="发送",
            command=self.send_message,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            self.input_frame,
            text="清空",
            command=self.clear_history
        ).pack(side=tk.LEFT)

        # 状态栏
        self.status_bar = ttk.Label(
            self.master,
            relief=tk.FLAT,
            anchor=tk.W,
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 样式配置
        self.history_area.tag_config('user', foreground='#092642')
        self.history_area.tag_config('assistant', foreground='#000000')
        self.history_area.tag_config('error', foreground='#761717')

    def show_model_manager(self):
        """显示模型管理界面"""
        # 隐藏主界面组件
        self.control_frame.pack_forget()
        self.history_area.pack_forget()
        self.input_frame.pack_forget()

        # 创建模型管理面板
        self.model_manager_frame = ttk.Frame(self.master)

        # 管理界面标题
        ttk.Label(self.model_manager_frame, text="模型管理", font=('Microsoft YaHei', 14)).pack(pady=10)

        # 返回按钮
        ttk.Button(
            self.model_manager_frame,text="返回聊天",command=self.return_to_chat).pack(side=tk.BOTTOM, pady=10)

        # 添加模型操作区域
        operation_frame = ttk.Frame(self.model_manager_frame)
        operation_frame.pack(fill=tk.X, pady=10)

        # 模型搜索/下载输入框
        ttk.Label(operation_frame, width=5).pack(side=tk.LEFT)
        self.new_model_var = tk.StringVar()
        ttk.Entry(operation_frame, textvariable=self.new_model_var, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            operation_frame,
            text="下载模型",
            command=self.download_model
        ).pack(side=tk.LEFT)

        # 模型列表
        self.model_tree = ttk.Treeview(
            self.model_manager_frame,
            columns=('hash','size', 'modified'),
            show='tree headings',
            selectmode='browse',
            style = 'Treeview'
        )
        self.model_tree.heading('#0', text='模型名称')
        self.model_tree.heading('hash', text='哈希值')
        self.model_tree.heading('size', text='大小')
        self.model_tree.heading('modified', text='更新时间')
        self.model_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 操作按钮
        btn_frame = ttk.Frame(self.model_manager_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="刷新列表", command=self.refresh_model_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="删除模型", command=self.delete_model).pack(side=tk.LEFT, padx=5)

        # 初始化模型列表
        self.refresh_model_list()

        # 显示管理界面
        self.model_manager_frame.pack(fill=tk.BOTH, expand=True)

    def return_to_chat(self):
        """返回聊天主界面"""
        self.model_manager_frame.destroy()
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        self.history_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.input_frame.pack(fill=tk.X, padx=5, pady=5)
        self.update_model_menu()  # 新增返回时刷新列表

    def update_status(self, message):
        self.status_bar.config(text=message)



    def append_to_history(self, text, role=None):
        self.history_area.config(state='normal')

        if role == 'user':
            self.history_area.insert(tk.END, "用户: ", 'user')
        elif role == 'assistant':
            self.history_area.insert(tk.END, "AI: ", 'assistant')

        self.history_area.insert(tk.END, text + "\n\n")
        self.history_area.config(state='disabled')
        self.history_area.see(tk.END)

    def clear_history(self):
        self.client.clear_history()
        self.history_area.config(state='normal')
        self.history_area.delete(1.0, tk.END)
        self.history_area.config(state='disabled')
        self.update_status("对话历史已清空")

    def send_message(self, event=None):
        try:
            new_model = self.model_var.get()
            self.client.model = new_model
            self.update_status(f"当前模型: {new_model}")
        except Exception as e:
            messagebox.showerror("错误", f"模型切换失败: {str(e)}")
        prompt = self.input_entry.get().strip()
        if not prompt:
            return

        self.input_entry.delete(0, tk.END)
        self.append_to_history(prompt, 'user')

        # 启动后台线程处理请求
        Thread(target=self.process_query, args=(prompt,)).start()

    def process_query(self, prompt):
        self.streaming = True
        self.current_response = ""

        try:
            # 获取生成参数
            temperature = round(self.temp_slider.get(), 1)

            # 流式响应处理
            response_stream = self.client.chat(
                prompt,
                stream=True,
                temperature=temperature
            )

            # 实时更新界面
            for chunk in response_stream:
                if chunk and chunk.strip():
                    self.current_response += chunk
                    self.master.after(10, self.update_stream_display)

            # 完成响应后整理格式
            self.master.after(10, self.finalize_response)

        except Exception as e:
            self.master.after(10, self.show_error, str(e))
        finally:
            self.streaming = False

    def update_stream_display(self):
        if not self.streaming:
            return

        # 删除最后一行临时内容
        self.history_area.config(state='normal')
        self.history_area.delete("end-2l linestart", tk.END)

        # 插入最新内容
        self.history_area.insert(tk.END, "AI: ", 'assistant')
        self.history_area.insert(tk.END, self.current_response + "▌")
        self.history_area.see(tk.END)
        self.history_area.config(state='disabled')

    def finalize_response(self):
        self.history_area.config(state='normal')
        self.history_area.delete("end-2l linestart", tk.END)
        self.history_area.insert(tk.END, "AI: ", 'assistant')
        self.history_area.insert(tk.END, self.current_response + "\n\n")
        self.history_area.config(state='disabled')
        self.update_status("响应完成 | 模型: " + self.client.model)

    def show_error(self, message):
        self.append_to_history(f"错误: {message}", 'error')
        self.update_status(f"错误: {message}")

    def refresh_model_list(self):
        """刷新模型列表"""
        self.model_tree.delete(*self.model_tree.get_children())
        try:
            models = self.client.get_installed_models()  # 需要OllamaClient实现此方法
            for model in models:
                self.model_tree.insert('', 'end',text=model['name'],
                    values=(
                        model.get('digest')[:12],
                        f"{model.get('size')/1024/1024:.1f} MB",  # 大小列
                        model.get('modified_at')[:16]  # 截取前16个字符（去除秒和时区）
                ))
            self.update_model_menu()  # 新增刷新后更新下拉菜单
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
            return

        model_name = self.model_tree.item(selected[0])['text']
        if messagebox.askyesno("确认", f"确定删除模型 {model_name} 吗？"):
            try:
                self.client.delete_model(model_name)
                messagebox.showinfo("成功", f"模型 {model_name} 已成功删除")
                self.refresh_model_list()
            except Exception as e:
                messagebox.showerror("错误", f"删除失败: {str(e)}")




if __name__ == "__main__":
    root = tk.Tk()
    app = OllamaGUI(root)
    root.mainloop()