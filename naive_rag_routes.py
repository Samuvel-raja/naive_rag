
from fastapi import APIRouter, UploadFile, File, HTTPException
from naive_rag import Naive_rag
import os

router = APIRouter()

rag = Naive_rag()

UPLOAD_DIR = "docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        file_path = os.path.join(UPLOAD_DIR, "sample.pdf")

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return {
            "message": "PDF uploaded successfully",
            "filename": "sample.pdf"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag")
async def rag_api(query:str):
    try:
        print("q",query)
        res=await rag.llm_brain(query)
        print(res)
        return {"msg":res}
    except Exception as e:
        print(e)