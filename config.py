from pydantic_settings import BaseSettings,SettingsConfigDict
from typing import Optional

class Config(BaseSettings):
   model_config=SettingsConfigDict(
     env_file=".env",
     env_ignore_empty=True,
     extra="ignore"
   )
   open_ai_api_key:str=""
   redis_url:str=""
   chunk_size:int=0
   chunk_overlap:int=0
   naive_docs_count:int=0
   qdrant_url:str=""
   collection_name:str=""

settings=Config()

