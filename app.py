import os
import sys

# 1. 路径自动对齐
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# 2. 清理代理
for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(var, None)

# --- 核心修正点 ---
# 检查环境变量。如果没有设置，则指向你电脑上真实的 3B 模型路径
if "ARCHITECT_BASE_MODEL" not in os.environ:
    # 填入你的真实路径：如：/data/ai-prompt/models/qwen/Qwen2.5-3B-Raw
    os.environ["ARCHITECT_BASE_MODEL"] = "/data/ai-prompt/models/qwen/Qwen2.5-3B-Raw"

try:
    import gradio as gr
    # 必须先设置好环境变量，再导入 engine
    from core.engine import architect, demo
except ImportError as e:
    print(f"环境缺失: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print(f"ARCHITECT 2.0 DEMO 启动中...")
    print(f"项目根目录: {ROOT_DIR}")
    print(f"基础模型路径: {os.environ.get('ARCHITECT_BASE_MODEL')}")
    
    # 针对 Gradio 6.0 的语法修正：theme 建议在 launch 中处理
    # app.py 结尾
demo.launch(
    server_name="0.0.0.0", 
    server_port=8888, 
    share=False,
    prevent_thread_lock=True 
)
