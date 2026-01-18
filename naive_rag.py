
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from config import Config
from langchain_community.document_loaders import PyPDFLoader
import asyncio

settings=Config()

class Naive_rag:
    def __init__(self):
        self.open_ai_api_key=settings.open_ai_api_key
        self.chunk_overlap=settings.chunk_overlap
        self.chunk_size=settings.chunk_size
        self.qdrant_url=settings.qdrant_url
        self.collection_name=settings.collection_name

    async def load_pdf(self):
        try:
            pdf_loader= PyPDFLoader("./docs/sample.pdf")
            docs=pdf_loader.load()
            return docs
        except Exception as e:
            print(e)
    async def split_into_chunks(self):
        docs=await self.load_pdf()
        try:
            text_splitter= RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks=text_splitter.split_documents(docs)
            return chunks
        except Exception as e:
            print(e)
    async def create_embeddings(self):
        try:
            embedding=OpenAIEmbeddings(
                api_key=self.open_ai_api_key,
            model="text-embedding-3-small",
            )
            return embedding
        except Exception as e:
            print(e)
    async def store_in_vectordb(self):
        try:
            chunks=await self.split_into_chunks()
            embeddings=await self.create_embeddings()

            vector_store=QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                url=self.qdrant_url,
                collection_name=self.collection_name
            )
            return vector_store
        except Exception as e:
            print(e)
    async def retriever(self,query):
        try:
            vector_store=await self.store_in_vectordb()
            retriever=vector_store.as_retriever(
                    search_type="similarity",
                  search_kwargs={"k": 4}
            )
            retrieved_data=await retriever.ainvoke(query)
            return retrieved_data

        except Exception as e:
            print(e)

    async def llm_brain(self,query):
        try:
            print("query",query)
            retrieved_data=await self.retriever(query)
            prompt=ChatPromptTemplate.from_template(
                """
                Answer the question {query} based on this data {context}

                Strictly answer only from this context 

                if it's not in context just say I dont know
                """
            )
            context = "\n\n".join(d.page_content for d in retrieved_data)
            llm=ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    max_retries=3,
                    api_key=settings.open_ai_api_key
                )
            chain=prompt | llm | StrOutputParser()
            response=await chain.ainvoke({"query": query,"context":context})
            return response
        except Exception as e:
            print(e)

      

naive_rag=Naive_rag()
# async def main():
#     res = await naive_rag.llm_brain("What is the capital of France?")
#     print(res)

# if __name__ == "__main__":
#     asyncio.run(main())







