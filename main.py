import os
import asyncio
import traceback
from typing import Optional
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel

# Global variables for lazy loading
_openai_client: Optional['OpenAIRepository'] = None
_supabase_client: Optional['SupabaseRepository'] = None
_rag_client: Optional['RAGRepository'] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI application starting up...")
    print(f"Environment: {'Production' if not os.path.exists('.env') else 'Development'}")
    yield
    print("FastAPI application shutting down...")


app = FastAPI(
    title="LLM RAG FastAPI",
    description="API para procesamiento de embeddings con OpenAI",
    version="1.0.0",
    lifespan=lifespan
)


# Health check endpoints
@app.get("/")
async def root():
    return {
        "message": "LLM RAG FastAPI is running",
        "status": "healthy",
        "version": "1.0.0",
        "embedding_provider": "openai"
    }


@app.get("/health")
async def health_check():
    try:
        health_status = {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "services": {}
        }

        # Check environment variables
        openai_key = os.getenv("OPENAI_API_KEY")
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")

        health_status["services"] = {
            "openai": "configured" if openai_key else "missing_key",
            "supabase": "configured" if (supabase_url and supabase_key) else "missing_config"
        }

        return health_status
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


# Lazy loading functions
async def get_openai_client():
    global _openai_client
    if _openai_client is None:
        try:
            print("Creating OpenAI client instance")
            from utils.repository.openai_repository import OpenAIRepository
            _openai_client = OpenAIRepository()
            print("OpenAI client created successfully")
        except Exception as e:
            print(f"Error creating OpenAI client: {e}")
            raise
    return _openai_client


async def get_supabase_client():
    global _supabase_client
    if _supabase_client is None:
        try:
            print("Creating Supabase client instance")
            from utils.repository.supabase_repository import SupabaseRepository
            _supabase_client = SupabaseRepository()
            print("Supabase client created successfully")
        except Exception as e:
            print(f"Error creating Supabase client: {e}")
            raise
    return _supabase_client


async def get_rag_client():
    global _rag_client
    if _rag_client is None:
        try:
            print("Creating RAG client instance with OpenAI embeddings")
            # FIXED: Correct import path
            from utils.repository.rag_repository import RAGRepository
            model_name = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-large")
            _rag_client = RAGRepository(model_name=model_name)
            print(f"RAG client created successfully with model: {model_name}")
        except Exception as e:
            print(f"Error creating RAG client: {e}")
            raise
    return _rag_client


# Request models
class CreateRequest(BaseModel):
    system: str
    prompt: str
    model: Optional[str] = None
    effort: Optional[str] = "low"


class QuestionRequest(BaseModel):
    topic: int
    prompt: str
    academy: int
    model: str = "gpt-4o"
    has4questions: bool = True
    num_of_q: int = 5


class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
    min_similarity: Optional[float] = 0.1


# API Routes - Embedded directly to avoid circular imports
@app.post("/api/create")
async def create_text(request: CreateRequest):
    try:
        client = await get_openai_client()
        response = client.generate_text(
            system=request.system,
            prompt=request.prompt,
            model=request.model,
            effort=request.effort
        )
        return {"response": response}
    except Exception as e:
        print(f"Error in create_text: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")


@app.post("/api/questions")
async def generate_questions(request: QuestionRequest):
    try:
        print(f"Starting question generation for topic {request.topic}")

        client = await get_openai_client()
        supabase_client = await get_supabase_client()
        rag = await get_rag_client()

        print(f"Processing prompt: '{request.prompt[:100]}...'")

        similar_documents = await rag.search_similar_documents(
            query=request.prompt,
            limit=5,
            min_similarity=0.1
        )

        documents_content = [doc.get('content', '') for doc in similar_documents if doc.get('content')]
        context = ' '.join(documents_content) if documents_content else ""

        print(f"RAG context length: {len(context)} characters")
        print(f"Found {len(similar_documents)} similar documents")

        response = await client.generate_questions(
            topic=request.topic,
            academy=request.academy,
            has4questions=request.has4questions,
            prompt=request.prompt,
            num_of_q=request.num_of_q,
            model=request.model,
            context=context
        )

        saved_count = 0
        if isinstance(response, list) and response:
            for question in response:
                try:
                    if hasattr(question, 'to_json_without_id'):
                        supabase_client.insert(
                            table="questions",
                            data=question.to_json_without_id()
                        )
                        saved_count += 1
                except Exception as e:
                    print(f"Error saving question to database: {e}")

            print(f"Saved {saved_count} questions to database")

        return {
            "questions": response,
            "context_used": len(context) > 0,
            "context_length": len(context),
            "similar_documents_found": len(similar_documents),
            "questions_saved": saved_count
        }

    except Exception as e:
        print(f"Error in generate_questions: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")


@app.post("/api/search")
async def search_documents(request: SearchRequest):
    try:
        rag = await get_rag_client()

        similar_documents = await rag.search_similar_documents(
            query=request.query,
            limit=request.limit,
            min_similarity=request.min_similarity
        )

        return {
            "query": request.query,
            "total_found": len(similar_documents),
            "documents": similar_documents
        }

    except Exception as e:
        print(f"Error in search_documents: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")


@app.get("/api/status")
async def api_status():
    try:
        services_status = {}

        try:
            await get_openai_client()
            services_status["openai"] = "available"
        except Exception as e:
            services_status["openai"] = f"error: {str(e)}"

        try:
            await get_supabase_client()
            services_status["supabase"] = "available"
        except Exception as e:
            services_status["supabase"] = f"error: {str(e)}"

        try:
            await get_rag_client()
            services_status["rag"] = "available"
        except Exception as e:
            services_status["rag"] = f"error: {str(e)}"

        return {
            "api_status": "running",
            "services": services_status
        }

    except Exception as e:
        print(f"Error in api_status: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error checking status: {str(e)}")


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"Unhandled exception: {exc}")
    traceback.print_exc()
    return {
        "error": "Internal server error",
        "detail": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else "An error occurred"
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    print(f"Starting server on {host}:{port} with log level {log_level}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
        timeout_keep_alive=300
    )