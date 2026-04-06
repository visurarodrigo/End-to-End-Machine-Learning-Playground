"""Router package for the FastAPI application."""

from app.routes.classification import router as classification_router
from app.routes.core import router as core_router
from app.routes.unsupervised import router as unsupervised_router

