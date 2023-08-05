from langchain.utilities import BingSearchAPIWrapper, DuckDuckGoSearchAPIWrapper
from configs.model_config import BING_SEARCH_URL, BING_SUBSCRIPTION_KEY
from fastapi import Body
from fastapi.responses import StreamingResponse
from configs.model_config import (llm_model_dict, LLM_MODEL, SEARCH_ENGINE_TOP_K, PROMPT_TEMPLATE)
from server.chat.utils import wrap_done
from server.utils import BaseResponse
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import json


def bing_search(text, result_len=SEARCH_ENGINE_TOP_K):
    if not (BING_SEARCH_URL and BING_SUBSCRIPTION_KEY):
        return [{"snippet": "please set BING_SUBSCRIPTION_KEY and BING_SEARCH_URL in os ENV",
                 "title": "env info is not found",
                 "link": "https://python.langchain.com/en/latest/modules/agents/tools/examples/bing_search.html"}]
    search = BingSearchAPIWrapper(bing_subscription_key=BING_SUBSCRIPTION_KEY,
                                  bing_search_url=BING_SEARCH_URL)
    return search.results(text, result_len)


def duckduckgo_search(text, result_len=SEARCH_ENGINE_TOP_K):
    search = DuckDuckGoSearchAPIWrapper()
    return search.results(text, result_len)


SEARCH_ENGINES = {"bing": bing_search,
                  "duckduckgo": duckduckgo_search,
                  }


def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs


def lookup_search_engine(
        query: str,
        search_engine_name: str,
        top_k: int = SEARCH_ENGINE_TOP_K,
):
    results = SEARCH_ENGINES[search_engine_name](query, result_len=top_k)
    docs = search_result2docs(results)
    return docs


def search_engine_chat(query: str = Body(..., description="用户输入", example="你好"),
                       search_engine_name: str = Body(..., description="搜索引擎名称", example="duckduckgo"),
                       top_k: int = Body(SEARCH_ENGINE_TOP_K, description="检索结果数量"),
                       ):
    if search_engine_name not in SEARCH_ENGINES.keys():
        return BaseResponse(code=404, msg=f"未支持搜索引擎 {search_engine_name}")

    async def search_engine_chat_iterator(query: str,
                                          search_engine_name: str,
                                          top_k: int,
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

        docs = lookup_search_engine(query, search_engine_name, top_k)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

        chain = LLMChain(prompt=prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = [
            f"""出处 [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n"""
            for inum, doc in enumerate(docs)
        ]

        async for token in callback.aiter():
            # Use server-sent-events to stream the response
            yield json.dumps({"answer": token,
                   "docs": source_documents})
        await task

    return StreamingResponse(search_engine_chat_iterator(query, search_engine_name, top_k),
                             media_type="text/event-stream")