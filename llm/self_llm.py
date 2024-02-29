'''
定义一个self llm类，这是基于langchain llm 框架的一个拓展。
这个类为不同的大语言模型api封装了一个统一的接口。
具体要实现的功能有：
1.封装和简化API调用: Self_LLM类的设计旨在简化与不同大语言模型API之间的交互。
通过提供一个统一的接口，开发者可以更容易地集成和切换不同的AI服务，而不需要针对每个服务编写特定的调用代码。

2.参数配置: 类中的属性允许开发者灵活配置API调用的关键参数，比如选择模型、设置请求超时时间和调整生成文本的温度系数等。
这些参数的配置使得开发者能够根据具体的应用场景和需求，调整模型的行为。

3.扩展性: 通过model_kwargs属性，开发者可以传递额外的、特定于某个模型的参数。
这提供了更大的灵活性，使得Self_LLM类可以适用于多种不同的模型和API，即使它们需要不同的参数集。
'''
# 导入需要的依赖
from langchain.llms.base import LLM
# python 库，用于数据注解和验证
from typing import Dict, Any, Mapping
from pydantic import Field
# 定义类
class Self_LLM(LLM):
    url: str = None
    model_name: str='gpt-3.5-turbo'
    request_timeout: float = 0.1
    api_key: str = None
    '''
    Field(default_factory=dict): 使用Field函数来定义这个字段的默认值。
    default_factory=dict意味着如果没有提供值，则默认使用空字典作为值。
    这对于确保在没有提供具体值时变量不会是None，而是一个空的字典，非常有用。
    '''
    model_kwargs: Dict[str,Any] = Field(default_factory=dict)
    # 定义方法
    @property #将函数作为属性传入函数参数,用于只读属性
    def _default_params(self)->Dict[str,Any]:
        normal_params = {
            'temperature':self.temperature,
            'request_timeout':self.request_timeout,
        }
        return(normal_params)
    @property
    def _identifying_params(self)-> Mapping[str,Any]:
        # get the identifying parameters
        return{**{"model_name":self.model_name},**self._default_params}




