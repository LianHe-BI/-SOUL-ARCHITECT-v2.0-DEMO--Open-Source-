import os, torch, json, time, re
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, set_peft_model_state_dict

class SoulArchitect:
    def __init__(self):
        self.root = os.path.dirname(os.path.abspath(__file__)) # 修正路径定位
        self.base_path = os.getenv("ARCHITECT_BASE_MODEL", "/data/ai-prompt/models/qwen/Qwen2.5-3B-Raw")
        self.core_file = "/data/return/models/binary_cores/ARCHITECT_V2_PRO.core"
        self.history_file = os.path.join(self.root, "history.json")
        self.model = None
        self.tokenizer = None

    def boot(self):
        try:
            print(f"激活原生 BF16 : {self.base_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_path, trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(
                self.base_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            from safetensors import safe_open
            with safe_open(self.core_file, framework="pt", device="cuda") as f:
                metadata = f.metadata()
                config_dict = json.loads(metadata.get("config", "{}"))
                lora_config = LoraConfig(**config_dict)
                tensors = {k: f.get_tensor(k) for k in f.keys()}
            self.model = PeftModel(base, lora_config)
            set_peft_model_state_dict(self.model, tensors)
            self.model.eval()
            print("准备就绪")
            return True
        except Exception as e:
            print(f"启动失败: {e}"); return False

    def generate_architect_v2(self, user_input):
        if not user_input.strip(): return "", self.render_history_text()
        start_time = time.time()
        try:
            sys_text = "<|im_start|>system\nYou are the Soul Architect. Output format:\nPositive: (English keywords)\nNegative: (English keywords)\nChinese: (Summary)\n<|im_end|>\n"
            user_text = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\nPositive:"
            ins = self.tokenizer(sys_text + user_text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                out = self.model.generate(
                    **ins, max_new_tokens=512, temperature=0.7, do_sample=True,
                    top_p=0.9, repetition_penalty=1.2, no_repeat_ngram_size=3,
                    eos_token_id=self.tokenizer.eos_token_id 
                )

            raw = self.tokenizer.decode(out[0][ins.input_ids.shape[1]:], skip_special_tokens=True)
            
            def _clean(text, default=""):
                t = re.sub(r'[^\x00-\x7F]+', '', text) 
                res = ", ".join([p.strip().title() for p in re.split(r'[,，\n]', t) if len(p.strip()) > 1])
                return res if len(res) > 3 else default

            p_m = re.search(r"(.*?)(?=Negative|Chinese|中文|$)", raw, re.S | re.I)
            p_en = _clean(p_m.group(1) if p_m else raw.split('\n')[0])
            hardcore = "cinematic lighting, depth of field, high fidelity, texture details, ray traced shadows, global illumination, photorealistic, raw photo style, film grain"
            final_pos = f"{p_en}, {hardcore}" if p_en else hardcore

            n_m = re.search(r"Negative\s*[:：]?(.*?)(?=Chinese|中文|$)", raw, re.S | re.I)
            base_neg = "blurry, low quality, distorted, ugly, watermark, text, lowres, monochrome, cartoon, illustration"
            n_extracted = _clean(n_m.group(1) if n_m else "", "")
            final_neg = f"{n_extracted}, {base_neg}" if n_extracted and "None" not in n_extracted else base_neg
            
            c_m = re.search(r"(?:Chinese|中文|摘要)\s*[:：]?(.*?)$", raw, re.S | re.I)
            cn_res = re.sub(r'[a-zA-Z0-9_]+', '', c_m.group(1) if c_m else "").strip(" :：")
            cn_res = cn_res if len(cn_res) > 5 else f"记录：{user_input}"

            self.save_history({"u": user_input, "c": cn_res, "p": final_pos})
            return self._render_html(cn_res, final_pos, final_neg, time.time() - start_time), self.render_history_text()
        except Exception as e:
            return f"<div style='color:red;'>异常: {str(e)}</div>", self.render_history_text()
            
            
    def _render_html(self, cn, pos, neg, dur):
        return f"""
        <div style="font-family:sans-serif;">
            <div style="color:#2a9d8f; font-weight:900; border-bottom:1px solid #2a9d8f; display:inline-block; font-size:0.75em; margin-bottom:10px;">ARCHITECT v2.0 | DEMO | {dur:.2f}S</div>
            <div style="background:#fff; padding:15px; border-radius:8px; border:1px solid #eee; margin-bottom:15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <div style="color:#111; font-size:1.15em; font-weight:500; line-height:1.5;">{cn}</div>
            </div>
            <div style="background:#1a1a1a; padding:20px; border-radius:8px;">
                <div style="margin-bottom:15px;">
                    <b style="color:#666; font-size:0.6em; font-weight:900; letter-spacing:1px;">POSITIVE PROMPT (RAW PHOTO)</b>
                    <div style="color:#00ffcc; font-family:monospace; font-size:0.95em; border-left:2px solid #00ffcc; padding-left:12px; margin-top:8px;">{pos}</div>
                </div>
                <b style="color:#666; font-size:0.6em; font-weight:900; letter-spacing:1px;">NEGATIVE PROMPT</b>
                <div style="color:#ff6b6b; font-family:monospace; font-size:0.85em; opacity:0.8; margin-top:5px;">{neg}</div>
            </div>
        </div>
        """

    def render_history_text(self):
        h = self.load_history()
        if not h: return "<p style='color:#999; text-align:center; padding-top:20px;'>— STANDBY —</p >"
        html = '<div style="max-height: 450px; overflow-y: auto;">'
        for i in h[::-1]:
            html += f"""
            <div style="margin-bottom:10px; border:1px solid #eee; border-radius:6px; padding:10px; background:#fff;">
                <div style="font-weight:900; color:#444; font-size:0.8em; border-left:3px solid #2a9d8f; padding-left:8px;">{i.get('u', '未知')}</div>
                <div style="color:#2a9d8f; font-size:0.75em; margin-top:4px; font-style:italic;">{i.get('c', '无回响')}</div>
            </div>
            """
        return html + '</div>'

    def load_history(self):
        if not os.path.exists(self.history_file): return []
        try:
            with open(self.history_file, "r", encoding="utf-8") as f: return json.load(f)
        except: return []

    def save_history(self, entry):
        h = self.load_history()
        h.append(entry)
        with open(self.history_file, "w", encoding="utf-8") as f: 
            json.dump(h[-50:], f, ensure_ascii=False, indent=2)

    def clear_history_logic(self):
        """执行物理删除并返回清空后的HTML"""
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
        return self.render_history_text()

# 启动流程
architect = SoulArchitect()
architect.boot()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("<div style='text-align:center; padding:15px; font-weight:900; font-size:1.3em;'>SOUL ARCHITECT v2.0 <span style='color:#2a9d8f; font-size:0.6em;'>DEMO</span></div>")
    with gr.Row():
        with gr.Column(scale=4):
            u_in = gr.Textbox(label="意境构思", lines=3, placeholder="输入你的构思...")
            with gr.Row():
                btn_gen = gr.Button("开始生成", variant="primary")
                btn_clr = gr.Button("清空文本")
            out_h = gr.HTML()
        with gr.Column(scale=2):
            gr.Markdown("### 历史存档")
            hist_h = gr.HTML(value=architect.render_history_text())
    
    # 逻辑绑定
    btn_gen.click(architect.generate_architect_v2, inputs=[u_in], outputs=[out_h, hist_h])
    u_in.submit(architect.generate_architect_v2, inputs=[u_in], outputs=[out_h, hist_h])
    
    # 清空逻辑：返回空输入、空输出展示,保留历史HTML
    btn_clr.click(
        fn=lambda: ("", ""), 
        inputs=[], 
        outputs=[u_in, out_h]
    )
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8888, share=False)
