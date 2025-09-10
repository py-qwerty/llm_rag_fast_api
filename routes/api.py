import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, Response  # ✅ Response para headers
from pydantic import BaseModel
from dotenv import load_dotenv

# Cargar .env
parent_dir = Path(__file__).parent.parent
env_path = parent_dir / ".env"
load_dotenv(dotenv_path=env_path)

# Imports del proyecto
from routes.root import read_root as rr
from middlewares.validateToken import auth_dependency

# ✅ Importa servicio y modelo tipado del RAG
from utils.repository.rag_repository import BaseResponse as RAGBaseResponse, RAGService, RAGConfig

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG_MODE = os.getenv("DEBUG", "FALSE").upper() == "TRUE"

def conditional_auth():
    if DEBUG_MODE:
        return lambda: None
    return auth_dependency

auth_dep = conditional_auth()

# ==================== MODELOS (solo los necesarios) ====================
class QueryRequest(BaseModel):
    query: str

# ⚠️ Eliminamos QueryResponse con metadatos en el cuerpo.
#    El endpoint devolverá EXACTAMENTE RAGBaseResponse.

class SearchResponse(BaseModel):
    response: str
    query_type: str
    processing_time: float
    timestamp: float
    results_count: int
    search_metadata: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5

# ==================== SERVICIO RAG SINGLETON ====================
_rag_service = None
_initialization_error = None

def get_rag_service():
    """Instancia del RAG con salida tipada estricta."""
    global _rag_service, _initialization_error

    if _rag_service is None and _initialization_error is None:
        try:
            logger.info("🚀 Inicializando RAG Service con OpenAI Agents SDK...")
            try:
                _rag_config = RAGConfig(
                    model_name="gpt-5",
                    embedding_model="text-embedding-3-large",
                    search_limit=10,
                    agent_timeout=1000,
                    response_model=RAGBaseResponse,  # ✅ salida estricta {response, reasoning}
                )
                _rag_service = RAGService(config=_rag_config)

                logger.info("✅ RAG Service inicializado")
            except ImportError as e1:
                raise Exception(f"No se pudo importar el servicio RAG: {e1}")
        except Exception as e:
            _initialization_error = str(e)
            logger.error(f"❌ Error crítico inicializando RAG Service: {e}")
            raise

    return _rag_service

# ==================== ROUTER ====================
router = APIRouter()

# ==================== ENDPOINTS ====================

@router.post("/query", response_model=RAGBaseResponse)  # ✅ SOLO el modelo del agente
async def process_query(
    request: QueryRequest,
    response: Response,                # ✅ para enviar metadatos en headers
    user=Depends(auth_dep),
):
    """
    Devuelve EXACTAMENTE { response, reasoning }.
    Metadatos -> headers: X-Processing-Time, X-Timestamp.
    """
    start = time.time()

    try:
        # Servicio
        rag_service = get_rag_service()
        if not rag_service:
            raise HTTPException(status_code=503, detail="Servicio RAG no disponible. Intente más tarde.")

        # Validación
        query = (request.query or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query vacía")
        if len(query) > 5000:
            raise HTTPException(status_code=400, detail="Query demasiado larga")

        logger.info(f"🔍 Procesando con Agents SDK: {query[:100]}...")

        # Ejecutar
        result = await rag_service.process_query(query)

        # ✅ Normalizar a RAGBaseResponse
        if isinstance(result, RAGBaseResponse):
            final = result
        elif isinstance(result, dict):
            final = RAGBaseResponse.model_validate(result)
        elif isinstance(result, str):
            final = RAGBaseResponse(response=result, reasoning="")
        else:
            final = RAGBaseResponse(response=str(result), reasoning="")

        # ✅ Metadatos SOLO en headers (no en el JSON)
        response.headers["X-Processing-Time"] = f"{time.time() - start:.3f}"
        response.headers["X-Timestamp"] = str(time.time())

        return final

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Error procesando query: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.post("/search", response_model=SearchResponse)
async def process_search(
    request: SearchRequest,
    user=Depends(auth_dep),
):
    """
    Búsqueda directa (con metadatos en el cuerpo porque NO está sujeta al formato legal estricto).
    """
    start_time = time.time()

    try:
        rag_service = get_rag_service()
        if rag_service is None:
            raise HTTPException(status_code=503, detail="Servicio RAG no disponible")

        query = (request.query or "").strip()
        limit = request.limit or 5
        if not query:
            raise HTTPException(status_code=400, detail="Query vacía")

        logger.info(f"🔍 Búsqueda directa con SDK: {query[:100]}...")

        search_results = await rag_service.search_documents_directly(query, limit)
        processing_time = time.time() - start_time

        if search_results:
            service_type = getattr(rag_service, 'service_type', 'production')
            service_prefix = " (Mock)" if service_type == "mock" else ""
            response_text = f"📚 **RESULTADOS DE BÚSQUEDA{service_prefix}** ({len(search_results)} documentos encontrados)\n\n"
            for i, result in enumerate(search_results, 1):
                content_preview = result.get('content', '')[:150]
                similarity = result.get('similarity', 0.0)
                doc_id = result.get('document_id', 'N/A')
                law_title = result.get('law_title', 'Sin título')
                response_text += (
                    f"**{i}. {law_title}** (ID: {doc_id})\n"
                    f"📄 {content_preview}...\n"
                    f"🎯 Relevancia: {similarity:.1%}\n\n"
                )
        else:
            response_text = (
                "❌ **NO SE ENCONTRARON DOCUMENTOS**\n\n"
                "**Sugerencias:**\n"
                "• Usa términos más específicos\n"
                "• Incluye palabras clave legales\n"
                "• Verifica la ortografía\n"
                "• Prueba con sinónimos\n\n"
                "**Ejemplo:** \"procedimiento denuncia robo\" o \"artículo sanción administrativa\""
            )

        search_metadata = {
            "total_results": len(search_results),
            "search_limit": limit,
            "processing_method": "agents_sdk_tools",
            "embedding_used": True,
            "service_type": getattr(rag_service, 'service_type', 'production')
        }

        return SearchResponse(
            response=response_text,
            query_type="search",
            processing_time=processing_time,
            timestamp=time.time(),
            results_count=len(search_results),
            search_metadata=search_metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Error en búsqueda: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.get("/info")
async def system_info(user=Depends(auth_dep)):
    """Info del sistema (no afecta al formato del agente)."""
    try:
        service_status = "active"
        service_type = "unknown"
        service_error = None

        try:
            rag_service = get_rag_service()
            if rag_service:
                service_type = getattr(rag_service, 'service_type', 'production')
        except Exception as e:
            service_status = "error"
            service_error = str(e)

        return {
            "sistema": "RAG con OpenAI Agents SDK",
            "version": "v2.0-agents-sdk",
            "estado": service_status,
            "service_type": service_type,
            "initialization_error": _initialization_error,
            "service_error": service_error,
            "sdk_info": {
                "framework": "OpenAI Agents SDK",
                "features": [
                    "Handoffs automáticos",
                    "Structured outputs",
                    "Guardrails de seguridad",
                    "Function tools",
                    "Tracing integrado"
                ]
            },
            "agentes": {
                "coordinador": "Smart Coordinator",
                "especialista_legal": "Legal Analysis Specialist",
                "asistente_general": "General Police Assistant"
            },
            "configuracion": {
                "debug_mode": DEBUG_MODE,
                "jwt_protection": "DESACTIVADA" if DEBUG_MODE else "ACTIVADA",
                "embedding_provider": "openai",
                "modelo_principal": "gpt-5",
                "embedding_model": "text-embedding-3-large"
            }
        }
    except Exception as e:
        logger.exception(f"❌ Error obteniendo info: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/agents/status")
async def agents_status(user=Depends(auth_dep)):
    """Estado de agentes (no afecta al formato del agente)."""
    try:
        agent_status = {
            "agents_active": False,
            "sdk_version": "optimized",
            "service_available": False,
            "initialization_error": _initialization_error
        }

        try:
            rag_service = get_rag_service()
            if rag_service:
                agent_status.update({
                    "agents_active": True,
                    "service_available": True,
                    "service_type": getattr(rag_service, 'service_type', 'production')
                })
        except Exception as e:
            agent_status["service_error"] = str(e)

        return {
            "status": "active" if agent_status["service_available"] else "error",
            "timestamp": time.time(),
            "capabilities": [
                "Query classification",
                "Legal document analysis",
                "General assistance",
                "Security validation"
            ]
        }
    except Exception as e:
        logger.exception(f"❌ Error obteniendo estado: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/")
def read_root():
    return rr()
