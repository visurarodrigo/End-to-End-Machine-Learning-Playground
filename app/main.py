from fastapi import FastAPI

from app.routes.classification import router as classification_router
from app.routes.core import router as core_router
from app.routes.unsupervised import router as unsupervised_router


app = FastAPI()

app.include_router(core_router)
app.include_router(classification_router)
app.include_router(unsupervised_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
