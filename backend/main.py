"""
FastAPI Application Entry Point
=================================
Slim application setup that mounts modular routers.

WHAT CHANGED from the original:
- All business logic extracted to services/ and routers/
- main.py is now purely application setup (middleware, routers, lifespan)
- This follows the FastAPI "bigger applications" pattern:
  https://fastapi.tiangolo.com/tutorial/bigger-applications/

PRESERVED:
- Same CORS configuration
- Same endpoints (POST /upload, POST /chat)
- Same behavior — users won't notice any difference

ADDED:
- POST /feedback — user feedback submission
- GET  /analytics/* — dashboard data endpoints
- GET  /health — health check (standard for production services)
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers import upload, chat, feedback, analytics

app = FastAPI(
    title="RAG Chatbot API",
    description="PDF Question-Answering System with Evaluation & Observability",
    version="2.0.0",
)

# ─── CORS Middleware ────────────────────────────────────────────────────
# allow_credentials=True is incompatible with allow_origins=["*"] per the
# CORS spec. Since we don't use cookies/auth, we drop credentials and use
# a pure wildcard — this guarantees CORS headers on every response.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Exception Handler ──────────────────────────────────────────
# Ensures unhandled errors return a proper JSON response instead of a bare
# 500 that loses CORS headers (which makes the browser mask the real error
# as a misleading "blocked by CORS policy" message).
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# ─── Mount Routers ─────────────────────────────────────────────────────
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(feedback.router)
app.include_router(analytics.router)


# ─── Health Check ───────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """
    Basic health check endpoint.
    In production, this would verify database connectivity, model
    availability, and return detailed status.
    """
    return {"status": "healthy", "version": "2.0.0"}
