"""第 2 周 Day 2：OutputParser 结构化输出"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()


# ===== Demo 1：JsonOutputParser（简单版）=====
def json_parser_demo():
    print("=== Demo 1：JSON 输出解析 ===")

    llm = ChatOpenAI(
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-4-flash",
        temperature=0.3
    )

    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_template(
        "请提取以下句子中的信息，以 JSON 格式输出，包含字段：city, temperature, weather。\n"
        "句子：{sentence}\n"
        "输出格式：{format_instruction}"
    )

    chain = prompt | llm | parser

    result = chain.invoke({
        "sentence": "今天西安天气晴朗，气温25度",
        "format_instruction": parser.get_format_instructions()
    })

    print(f"解析结果: {result}")
    print(f"类型: {type(result)}")
    print(f"城市: {result.get('city')}, 温度: {result.get('temperature')}\n")


# ===== Demo 2：PydanticOutputParser（强类型版，推荐）=====
class WeatherInfo(BaseModel):
    city: str = Field(description="城市名")
    temperature: int = Field(description="温度，整数")
    weather: str = Field(description="天气状况，如晴/雨/多云")
    humidity: int = Field(description="湿度百分比，整数")


def pydantic_parser_demo():
    print("=== Demo 2：Pydantic 强类型解析 ===")

    llm = ChatOpenAI(
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-4-flash",
        temperature=0.3
    )

    parser = JsonOutputParser(pydantic_object=WeatherInfo)

    prompt = ChatPromptTemplate.from_template(
        "提取句子中的天气信息。\n"
        "句子：{sentence}\n"
        "格式要求：{format_instruction}"
    )

    chain = prompt | llm | parser

    result = chain.invoke({
        "sentence": "上海今天多云，气温28度，湿度65%",
        "format_instruction": parser.get_format_instructions()
    })

    pydan = WeatherInfo(**result)
    print(f"解析结果: {pydan}")
    print(f"城市: {pydan.city}, 温度: {pydan.temperature}°C")
    print(f"天气: {pydan.weather}, 湿度: {pydan.humidity}%\n")


# ===== Demo 3：Few-shot Prompt（给例子让 AI 学格式）=====
def few_shot_demo():
    print("=== Demo 3：Few-shot 示例 ===")
    print("debug now!!")

    llm = ChatOpenAI(
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-4-flash",
        temperature=0.5
    )

    few_shot_prompt = ChatPromptTemplate.from_messages([
        ("human", "输入：西安，晴，22度"),
        ("ai", '{{"city": "西安", "weather": "晴", "temperature": 22}}'),#double braces 转义
        ("human", "输入：广州，雨，30度"),
        ("ai", '{{"city": "广州", "weather": "雨", "temperature": 30}}'),
        ("human", "输入：{input}")
    ])


    chain = few_shot_prompt | llm | JsonOutputParser()

    result = chain.invoke({
        "input":"深圳，多云，27度"
    })

    print(f"Few-shot 结果: {result}\n")


if __name__ == "__main__":
    json_parser_demo()
    pydantic_parser_demo()
    few_shot_demo()
    print("=== Day 2 完成 ===")