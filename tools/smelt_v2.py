import os, torch, json, datetime
from safetensors.torch import save_file

def smelt_pro_bf16(lora_dir, output_path):
    print(f"BF16启用中 ...")
    
    # 加载权重
    st_path = os.path.join(lora_dir, "adapter_model.safetensors")
    from safetensors.torch import load_file as load_st
    weights = load_st(st_path)

    # 保持BF16
    bf16_weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}

    with open(os.path.join(lora_dir, "adapter_config.json"), "r") as f:
        config_data = json.load(f)
    
    metadata = {
        "format": "ARCHITECT_V2_PRO_BF16",
        "precision": "BF16",
        "config": json.dumps(config_data),
        "timestamp": str(int(datetime.datetime.now().timestamp()))
    }

    save_file(bf16_weights, output_path, metadata=metadata)
    print(f"BF16 核心已就位: {output_path}")

if __name__ == "__main__":
    smelt_pro_bf16("/data/return/models/lora/ARCHITECT_v2_PRO_MODE", "/data/return/models/binary_cores/ARCHITECT_V2_PRO.core")
