from fastapi import FastAPI
from confluence.endpoints import router as confluence_router

app = FastAPI()

# Include the routes with prefix `/api`
app.include_router(confluence_router, prefix="/api", tags=["Confluence"])
