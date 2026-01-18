
from fastapi import FastAPI
from naive_rag_routes import router as naive_router

app=FastAPI(title="Naive rag application")


app.include_router(naive_router)

















