import json
from dataclasses import dataclass,field
from typing import Optional

@dataclass
class ModelConfig:
    model:str = "glm-4-flash"
    temperature:float = 0.7
    max_tokens:int = 512
    top_p:float = 0.9

    @classmethod
    def from_json(cls,path:str)->"ModelConfig":
        """从JSON文件加载配置"""
        with open(path,"r",encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)#用path的配置文件生成一个ModelConfig对象

    def to_json(self,path:str)->None:
        """保存到JSON文件"""
        with open(path,"w",encoding="utf-8") as f:
            json.dump(self.__dict__,f,ensure_ascii=False,indent=2)

    def override(self,**kwargs)->"ModelConfig":
        """用kwargs覆盖配置，返回新实例（不修改原值）"""
        for key,value in kwargs.items():
            if hasattr(self,key):
                setattr(self,key,value)
        return self#用kwargs生成一个ModelConfig对象
