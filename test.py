# import wandb
# import pandas as pd

# wandb.login()
# YOUR_PROJECT = "Search"
# YOUR_ENTITY = "hyn-t-d-lemon-1900-university-of-electronic-science-and-"

# api = wandb.Api()
# runs = api.runs("peterjin/Search-R1-v0.3")

# for run in runs:
#     # 使用 scan_history() 获取完整数据（包括所有step）
#     history = list(run.scan_history())  # 关键修改点
#     config = run.config
#     summary = run.summary
    
#     with wandb.init(project=YOUR_PROJECT, entity=YOUR_ENTITY, name=run.name) as new_run:
#         wandb.config.update(config)
        
#         # 重新记录数据
#         for row in history:
#             wandb.log(row)
        
#         # 复制summary指标（包含val等最终指标）
#         for key, value in summary.items():
#             if isinstance(value, (int, float)):
#                 wandb.run.summary[key] = value
        
#         print(f"Copied run: {run.name}")

import wandb
import pandas as pd

wandb.login()
YOUR_PROJECT = "Search"
YOUR_ENTITY = "hyn-t-d-lemon-1900-university-of-electronic-science-and-"

api = wandb.Api()
runs = api.runs("peterjin/Search-R1-v0.3")

for run in runs:
    history = list(run.scan_history())
    config = run.config
    summary = run.summary
    
    print(f"\n=== Run: {run.name} ===")
    
    with wandb.init(project=YOUR_PROJECT, entity=YOUR_ENTITY, name=run.name) as new_run:
        if "3b" in run.name:
            wandb.config.update(config)
            
            # 重新记录历史数据
            for row in history:
                wandb.log(row)
            
            # ===== 修复：安全地处理 summary =====
            def flatten_dict(d, parent_key='', sep='/'):
                """将嵌套字典展平"""
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)
            
            # 先转换为普通 dict，避免 wandb summary 对象的特殊行为
            summary_dict = dict(summary)
            flat_summary = flatten_dict(summary_dict)
            
            # 安全地设置 summary
            for key, value in flat_summary.items():
                try:
                    # 只处理简单数值类型
                    if isinstance(value, (int, float)):
                        wandb.run.summary[key] = value
                    elif isinstance(value, str):
                        wandb.run.summary[key] = value
                    # 跳过其他复杂类型（包括 tensor、Image 等）
                except Exception as e:
                    print(f"  Skip {key}: {e}")
            
            # 从 history 中提取 val 数据
            for row in history:
                for key, value in row.items():
                    if 'val' in str(key).lower() and isinstance(value, (int, float)):
                        # 记录到 summary（会覆盖，保留最后值）
                        wandb.run.summary[key] = value
            
            print(f"Copied: {run.name}")
            print(f"Summary keys: {list(wandb.run.summary.keys())}")
        else:
            pass