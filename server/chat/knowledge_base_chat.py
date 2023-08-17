from fastapi import Body, Request
from fastapi.responses import StreamingResponse
from configs.model_config import (llm_model_dict, LLM_MODEL, PROMPT_TEMPLATE,
                                  VECTOR_SEARCH_TOP_K)
from server.chat.utils import wrap_done
from server.utils import BaseResponse
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBService, KBServiceFactory
import json
import os
from urllib.parse import urlencode
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.prompts import MessagesPlaceholder
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent

def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                        top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                        history: List[History] = Body([],
                                                      description="历史对话",
                                                      examples=[[
                                                          {"role": "user",
                                                           "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                          {"role": "assistant",
                                                           "content": "虎头虎脑"}]]
                                                      ),
                        stream: bool = Body(False, description="流式输出"),
                        local_doc_url: bool = Body(False, description="知识文件返回本地路径(true)或URL(false)"),
                        request: Request = None,
                        ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History(**h) if isinstance(h, dict) else h for h in history]

    async def knowledge_base_chat_iterator(query: str,
                                           kb: KBService,
                                           top_k: int,
                                           history: Optional[List[History]],
                                           ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        model = ChatOpenAI(
            streaming=True,
            verbose=True,
            callbacks=[callback],
            openai_api_key=llm_model_dict[LLM_MODEL]["api_key"],
            openai_api_base=llm_model_dict[LLM_MODEL]["api_base_url"],
            model_name=LLM_MODEL
        )
        
        def configure_retriever():
            return kb.to_langchain_receiver(top_k=top_k)
        
        tool = create_retriever_tool(
            configure_retriever(),
            "search_langsmith_docs",
            "Searches and returns documents regarding LangSmith. LangSmith is a platform for debugging, testing, evaluating, and monitoring LLM applications. You do not know anything about LangSmith, so if you are ever asked about LangSmith you should use this tool.",
        )
        
        tools = [tool]
        message = SystemMessage(
            content=(
                    f"You are a helpful chatbot who is tasked with answering questions about {knowledge_base_name}, "
                    f"Unless otherwise explicitly stated, it is probably fair to assume that questions are about {knowledge_base_name}, "
                    "If there is any ambiguity, you probably assume they are about that."
                )
        )
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
        )
        agent = OpenAIFunctionsAgent(llm=model, tools=tools, prompt=prompt)
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
        )
        # agent_executor = create_conversational_retrieval_agent(model, tools, verbose=True)
        
        

        # chat_prompt = ChatPromptTemplate.from_messages(
        #     [i.to_msg_tuple() for i in history] + [("human", PROMPT_TEMPLATE)])

        # chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.

        task = asyncio.create_task(wrap_done(
            agent_executor.acall(inputs={"input": query, "history":[i.to_langchain_message() for i in history] }),
            callback.done),
        )

        # source_documents = []
        # for inum, doc in enumerate(docs):
        #     filename = os.path.split(doc.metadata["source"])[-1]
        #     if local_doc_url:
        #         url = "file://" + doc.metadata["source"]
        #     else:
        #         parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name":filename})
        #         url = f"{request.base_url}knowledge_base/download_doc?" + parameters
        #     text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
        #     source_documents.append(text)

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token,
                                  "docs": ""},
                                 ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": ""},
                             ensure_ascii=False)

        await task

    return StreamingResponse(knowledge_base_chat_iterator(query, kb, top_k, history),
                             media_type="text/event-stream")
