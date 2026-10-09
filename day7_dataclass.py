from src.config import ModelConfig

#从JSON加载
cfg = ModelConfig.from_json("config.json")
print(f"模型：{cfg.model},温度：{cfg.temperature}")

#覆盖配置，覆盖cfg实例中的属性值
new_cfg = cfg.override(model="glm-4-plus", temperature=0.3)
print(f"新配置：{new_cfg.model},{new_cfg.temperature}")

#保存
new_cfg.to_json("config_backup.json")
print("备份已保存")