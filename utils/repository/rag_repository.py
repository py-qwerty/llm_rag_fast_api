import logging
import asyncio
import time
import inspect
from typing import List, Dict, Any, Optional, Union, Type, Tuple
from dataclasses import dataclass
import re

from agents import Agent, Runner, function_tool, AgentOutputSchema
from pydantic import BaseModel, Field, ConfigDict

from utils.repository.supabase_repository import SupabaseRepository
from utils.services.embedding_service import EmbeddingService
from utils.services.vector_search import VectorSearchService
from utils.services.legalSearchService import LegalSearchService
from utils.models.law_item_response import LawItemSimplify

# Configurar logging para suprimir logs de httpx pero mantener los nuestros
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# =========================
# Modelo de metadata
# =========================
class QueryMetadata(BaseModel):
    """Metadata de la consulta procesada"""
    processing_time_seconds: float = Field(description="Tiempo de procesamiento en segundos")
    law_identifiers_used: List[str] = Field(description="Lista de identificadores de ley utilizados",
                                            default_factory=list)
    item_ids_used: List[int] = Field(description="Lista de IDs de artículos utilizados", default_factory=list)
    tools_used: List[str] = Field(description="Herramientas de búsqueda utilizadas", default_factory=list)
    agent_calls: int = Field(description="Número de llamadas a agentes", default=0)
    handoffs_detected: int = Field(description="Número de handoffs detectados", default=0)
    documents_found: int = Field(description="Número total de documentos encontrados", default=0)

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"additionalProperties": False}
    )


# =========================
# Fallbacks de instrucciones
# =========================
_DEFAULT_LEGAL_PROMPT = """
Eres un especialista en derecho español con acceso a una base de datos legal.
Dispones de DOS herramientas de búsqueda:

1) search_legal_documents → búsquedas semánticas (embeddings).
2) search_legal_documents_hybrid → buscador SQL plano por texto.

PARÁMETROS DE BÚSQUEDA:
- El límite mínimo configurado es {min_search_limit}, pero PUEDES usar límites más altos según la complejidad:
  * Para consultas simples o específicas: usa entre {min_search_limit}-15 documentos
  * Para análisis complejos o comparativos: usa entre 15-25 documentos  
  * Para investigaciones exhaustivas: usa hasta 30-40 documentos
- Ajusta el límite según la naturaleza de la consulta y la cantidad de información necesaria

OBJETIVO CRÍTICO:
- En la salida final, bajo el bloque '--- TEXTOS LEGALES CONSULTADOS ---',
  debes incluir citas LITERALES EXACTAS del campo `content` devuelto por las herramientas.
- NO resumas, NO reescribas, NO corrijas ortografía, NO normalices comillas ni puntuación.
- Si necesitas acotar, selecciona el fragmento exacto RELEVANTE, pero sin alterar sus caracteres.
- Si NO hay contenido literal disponible, indica claramente que no hay texto fuente disponible.

ESTRATEGIA DE BÚSQUEDA:
- Si hay referencias concretas (p. ej., "Artículo 123", "Ley 10/1995"), usa search_legal_documents_hybrid.
- Para consultas conceptuales, usa search_legal_documents.
- Puedes usar ambas y combinar resultados. Evita duplicados.
- Considera hacer múltiples búsquedas con diferentes límites si la primera no es suficiente.

FORMATO DE SALIDA (JSON estricto del modelo Pydantic):
{{
  "response": "<tu explicación en lenguaje natural>\\n\\n--- TEXTOS LEGALES CONSULTADOS ---\\n<citas literales exactas, tal cual devueltas por las tools>",
  "reasoning": "Enumera búsquedas realizadas (qué tool, qué límite y por qué), documentos usados y por qué la interpretación es correcta."
}}
No añadas campos extra ni metadatos.
"""

_DEFAULT_GENERAL_PROMPT = """
Eres un asistente que responde SIEMPRE con un JSON que cumpla el modelo Pydantic requerido:
{
  "response": "<respuesta natural>",
  "reasoning": "breve justificación/metodología usada"
}
No añadas campos extra.
"""

_DEFAULT_COORDINATOR_PROMPT = """
Eres el coordinador. Decide si la consulta requiere análisis jurídico (deriva al especialista legal)
o una respuesta general. Exige que la salida final sea el JSON EXACTO del modelo de salida (sin campos extra).

IMPORTANTE: El especialista legal puede ajustar los límites de búsqueda según la complejidad:
- Consultas simples: {min_search_limit}-15 documentos
- Consultas complejas: 15-30 documentos  
- Investigaciones exhaustivas: hasta 40 documentos
"""


# =========================
# Modelo de salida Pydantic
# =========================
class BaseResponse(BaseModel):
    response: str = Field(description="La respuesta principal")
    reasoning: str = Field(description="Razonamiento de por qué se dio la respuesta", default="")
    citations: List[LawItemSimplify] = Field(
        description="Citas literales exactas de los documentos consultados, con el contenido y urls",
        default_factory=list)
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"additionalProperties": False}
    )


# Variable global para acceder al límite mínimo desde las tools
_MIN_SEARCH_LIMIT = 10


# ====================================================================
# TOOL: búsqueda vectorial (embeddings) — DEVUELVE CONTENIDO LITERAL
# ====================================================================
@function_tool
async def search_legal_documents(query: str, limit: Optional[int] = None) -> str:
    """
    Busca documentos legales usando embeddings vectoriales.

    Args:
        query: Consulta de búsqueda
        limit: Número de documentos a retornar. Si no se especifica, usa el mínimo configurado.
               Puedes usar límites más altos según la complejidad:
               - Consultas simples: 10-15 documentos
               - Consultas complejas: 15-30 documentos
               - Investigaciones exhaustivas: hasta 40 documentos

    Devuelve SIEMPRE contenido literal (campo `content`) SIN recortes ni modificaciones.
    """
    try:
        # Usar el límite proporcionado o el mínimo configurado
        effective_limit = limit if limit is not None else _MIN_SEARCH_LIMIT
        # Asegurar que el límite sea al menos el mínimo
        effective_limit = max(effective_limit, _MIN_SEARCH_LIMIT)

        logger.info(f"🔍 EMBEDDINGS: Iniciando búsqueda semántica para: '{query}' (límite: {effective_limit})")

        embedding_service = EmbeddingService(provider="openai", model_name="text-embedding-3-large")
        vector_search_service = VectorSearchService()

        logger.info("📡 EMBEDDINGS: Generando embedding para la consulta...")
        query_embedding = await embedding_service.generate_embedding(query)

        logger.info(f"🎯 EMBEDDINGS: Buscando vectores similares (umbral=0.2, límite={effective_limit})")
        search_results = await vector_search_service.search_similar_vectors(
            embedding=query_embedding,
            limit=effective_limit,
            similarity_threshold=0.2
        )

        if not search_results:
            logger.warning(f"⚠️ EMBEDDINGS: Sin resultados para: {query}")
            return f"No se encontraron documentos relevantes para: {query}"

        # Salida literal: nada de previews ni recortes
        lines: List[str] = [f"DOCUMENTOS ENCONTRADOS (Embeddings) para '{query}' (límite usado: {effective_limit}):",
                            ""]
        for i, item in enumerate(search_results[:effective_limit], 1):
            title = item.law_title or item.item_title or "Documento Legal"
            item_id = item.item_id
            law_identifier = item.law_identifier or ""
            url = item.url or ""
            item_title = item.item_title or "N/A"
            law_unic_id = item.law_unic_id or "N/A"
            content = item.content or ""

            lines += [
                f"{i}.  law title **{title}**",
                f"law_id  unic: {law_unic_id}",
                f"item_title  : {item_title}",
                f"; law_item_id {item_id}",
                (f";Identificador del BOE de ley: {law_identifier}" if law_identifier else ""),
                (f";URL: {url}" if url else ""),
                "   CONTENIDO:",
                content,  # ← CONTENIDO LITERAL COMPLETO
                ""
            ]
            logger.info(f'########----------{item.law_title} --> {item.item_title}----------########')

        out = "\n".join([ln for ln in lines if ln is not None])
        logger.info(
            f"✅ EMBEDDINGS: Retornando {len(search_results[:effective_limit])} documentos al agente (contenido literal)")
        return out

    except Exception as e:
        error_msg = f"Error buscando documentos (embeddings): {str(e)}"
        logger.error(f"❌ EMBEDDINGS: {error_msg}")
        return error_msg


# ====================================================================
# TOOL: búsqueda SQL plana — DEVUELVE CONTENIDO LITERAL
# ====================================================================
@function_tool
async def search_legal_documents_hybrid(query: str, law_id: Optional[int] = None, limit: Optional[int] = None) -> str:
    """
    Buscador SQL plano por texto (el agente decide estrategia).

    Args:
        query: Consulta de búsqueda
        law_id: ID específico de ley para filtrar (opcional)
        limit: Número de documentos a retornar. Si no se especifica, usa el mínimo configurado.
               Puedes usar límites más altos según la complejidad:
               - Consultas simples: 10-15 documentos
               - Consultas complejas: 15-30 documentos
               - Investigaciones exhaustivas: hasta 40 documentos

    Devuelve SIEMPRE el contenido literal completo de los documentos encontrados (campo `content`).
    """
    try:
        # Usar el límite proporcionado o el mínimo configurado
        effective_limit = limit if limit is not None else _MIN_SEARCH_LIMIT
        # Asegurar que el límite sea al menos el mínimo
        effective_limit = max(effective_limit, _MIN_SEARCH_LIMIT)

        logger.info(f"🔎 SQL: Búsqueda plana para: '{query}' (law_id={law_id}, límite={effective_limit})")

        service = LegalSearchService(schema_name="law_frame", rpc_name="search_legal_content")
        # Soporta servicios síncronos o asíncronos de forma flexible
        maybe_rows = service.search(query=query, limit=effective_limit, include_deprecated=False, law_id=law_id)
        rows = await maybe_rows if inspect.isawaitable(maybe_rows) else maybe_rows

        if not rows:
            logger.warning(f"⚠️ SQL: Sin resultados para: {query}")
            return f"No se encontraron documentos relevantes para: {query}"

        lines: List[str] = [f"DOCUMENTOS ENCONTRADOS (SQL) para '{query}' (límite usado: {effective_limit}):", ""]
        logger.info(f"✅ SQL: {len(rows)} documentos, preparando salida literal...")

        for i, r in enumerate(rows[:effective_limit], 1):
            title = r.law_title or r.item_title or "Documento Legal"
            item_id = r.item_title or "N/A"
            law_identifier = r.law_identifier or ""
            url = r.url or ""
            item_title = r.item_title or "N/A"
            law_unic_id = r.law_unic_id or "N/A"
            content = r.content or ""

            lines += [
                f"{i}.  law title **{title}**",
                f"law_id  unic: {law_unic_id}",
                f"item_title  : {item_title}",
                f"; law_item unic id {item_id}",
                (f";Identificador del BOE de ley: {law_identifier}" if law_identifier else ""),
                (f";URL: {url}" if url else ""),
                "   CONTENIDO:",
                content,  # ← CONTENIDO LITERAL COMPLETO
                ""
            ]

            logger.info(f'########----------{r.law_title} --> {r.item_title}----------########')
        return "\n".join([ln for ln in lines if ln is not None])

    except Exception as e:
        logger.error(f"❌ SQL: Error en búsqueda: {e}")
        return f"Error en búsqueda: {e}"


# ============================================================
# Utilidades para prompts dinámicos desde Supabase (con fallback)
# ============================================================
def _get_prompts_from_supabase(supabase: SupabaseRepository, destination: str) -> tuple[Optional[str], Optional[str]]:
    """
    Obtiene (instructions, model) desde Supabase por destino.
    Función interna para manejo de prompts dinámicos.
    """
    try:
        result = supabase.select(table='gpt_prompts', limit=1, filters={'destination': destination})
        if result and len(result) > 0:
            return result[0].get('prompt_system', None), result[0].get('model', None)
        return (None, None)
    except Exception as e:
        logger.error(f"Error obteniendo prompt de Supabase para '{destination}': {e}")
        return (None, None)


# =================================
# Creación de agentes (todos STRICT)
# =================================
def _create_legal_specialist(
        supabase: SupabaseRepository,
        min_search_limit: int,
        output_format: Optional[Type[BaseModel]] = None
) -> Agent:
    """
    Agente especialista en análisis legal con salida estricta y herramientas de búsqueda.
    Función interna para configuración de agente legal.
    """
    logger.info("🏗️ Creando Legal Analysis Specialist...")
    instructions, model = _get_prompts_from_supabase(supabase, 'legal_search_and_cite')
    output_format = output_format or BaseResponse

    # Insertar el límite mínimo en las instrucciones
    final_instructions = instructions or _DEFAULT_LEGAL_PROMPT

    agent = Agent(
        name="Legal Analysis Specialist",
        instructions=final_instructions,
        model=model,
        tools=[search_legal_documents, search_legal_documents_hybrid],
        output_type=AgentOutputSchema(output_format, strict_json_schema=True)
    )
    logger.info(
        f"✅ Legal Analysis Specialist creado (modelo: {model}, límite mín: {min_search_limit}, herramientas: 2)")
    return agent


def _create_general_assistant(
        supabase: SupabaseRepository,
        output_format: Optional[Type[BaseModel]] = None
) -> Agent:
    """
    Agente general con salida estricta (mismo esquema).
    Función interna para configuración de agente general.
    """
    logger.info("🏗️ Creando General Assistant...")
    instructions, model = _get_prompts_from_supabase(supabase, 'emoji')
    output_format = output_format or BaseResponse

    agent = Agent(
        name="General Assistant",
        instructions=instructions or _DEFAULT_GENERAL_PROMPT,
        model=model,
        output_type=AgentOutputSchema(output_format, strict_json_schema=True)
    )
    logger.info(f"✅ General Assistant creado (modelo: {model})")
    return agent


def _create_coordinator(
        legal_agent: Agent,
        general_agent: Agent,
        supabase: SupabaseRepository,
        min_search_limit: int,
        output_format: Optional[Type[BaseModel]] = None
) -> Agent:
    """
    Coordinador con handoffs y salida estricta.
    Función interna para configuración del agente coordinador.
    """
    logger.info("🏗️ Creando Query Coordinator...")
    instructions, model = _get_prompts_from_supabase(supabase, 'coordinator')
    output_format = output_format or BaseResponse


    final_instructions = instructions or _DEFAULT_COORDINATOR_PROMPT

    agent = Agent(
        name="Query Coordinator",
        instructions=final_instructions,
        model=model,
        handoffs=[legal_agent, general_agent],
    )
    logger.info(f"✅ Query Coordinator creado (modelo: {model}, límite mín: {min_search_limit}, handoffs: 2)")
    return agent


# ======================
# Config y servicio RAG
# ======================
@dataclass
class RAGConfig:
    model_name: str = "gpt-5"
    embedding_model: str = "text-embedding-3-large"
    search_limit: int = 10  # Este es ahora el LÍMITE MÍNIMO
    agent_timeout: int = 45
    output_format: str = "structured"
    response_model: Optional[Type[BaseModel]] = None


class RAGService:
    """Servicio RAG con modelo de respuesta dinámico y validación estricta."""

    def __init__(self, config: Optional[RAGConfig] = None):
        logger.info("🚀 Inicializando RAG Service...")
        self.config = config or RAGConfig()
        self.response_model = self.config.response_model or BaseResponse

        # Establecer el límite mínimo global
        global _MIN_SEARCH_LIMIT
        _MIN_SEARCH_LIMIT = self.config.search_limit

        # Métricas simples (privadas por convención)
        self._agent_calls = 0
        self._handoffs_simulated = 0
        self._tools_used: List[str] = []
        self._service_type = "production"

        # Servicios base (privados)
        logger.info("🔧 Inicializando servicios base...")
        self._embedding_service = EmbeddingService(provider="openai", model_name=self.config.embedding_model)
        self._vector_search = VectorSearchService()

        # Agentes (privados)
        logger.info("👥 Configurando agentes...")
        self._setup_agents()

        logger.info(
            f"✅ RAG Service inicializado (modelo: {self.response_model.__name__}, límite mín: {self.config.search_limit})")

    def _setup_agents(self):
        """
        Configura los agentes con modelo dinámico y prompts fallback.
        Método privado para inicialización interna.
        """
        supabase = SupabaseRepository()
        self._legal_specialist = _create_legal_specialist(
            supabase,
            min_search_limit=self.config.search_limit,
            output_format=self.response_model
        )
        self._general_assistant = _create_general_assistant(supabase, output_format=self.response_model)
        self._coordinator = _create_coordinator(
            self._legal_specialist,
            self._general_assistant,
            supabase,
            min_search_limit=self.config.search_limit,
            output_format=self.response_model
        )
        logger.info("👥 Todos los agentes configurados correctamente")

    def _extract_metadata_from_result(self, result, processing_time: float) -> QueryMetadata:
        """
        Extrae metadata del resultado del agente y genera estadísticas.
        """
        metadata = QueryMetadata(
            processing_time_seconds=round(processing_time, 3),
            agent_calls=self._agent_calls,
            tools_used=list(set(self._tools_used[-10:])),  # Últimas 10 herramientas únicas
        )

        try:
            # Analizar handoffs
            if hasattr(result, 'messages') and len(result.messages) > 2:
                metadata.handoffs_detected = len(result.messages) - 2
                self._handoffs_simulated += metadata.handoffs_detected

            # Extraer identificadores de ley del contenido del resultado
            result_content = str(result)

            # Buscar law_id en el contenido
            law_id_pattern = r'law_id:\s*([^\s,\n]+)'
            law_ids = re.findall(law_id_pattern, result_content)

            # Buscar item_id en el contenido
            item_id_pattern = r'item_id:\s*(\d+)'
            item_ids = [int(x) for x in re.findall(item_id_pattern, result_content)]

            # Buscar identificadores de ley (formato "Ley XX/XXXX", "Real Decreto XX/XXXX", etc.)
            law_identifier_pattern = r'(Ley\s+\d+/\d+|Real\s+Decreto\s+\d+/\d+|Real\s+Decreto-ley\s+\d+/\d+|Decreto\s+\d+/\d+)'
            law_identifiers = re.findall(law_identifier_pattern, result_content, re.IGNORECASE)

            # Contar documentos encontrados y extraer límites usados
            doc_count_pattern = r'DOCUMENTOS ENCONTRADOS.*?límite usado:\s*(\d+)'
            doc_matches = re.findall(doc_count_pattern, result_content)
            metadata.documents_found = len(doc_matches)

            # Log de límites usados para debugging
            if doc_matches:
                limits_used = [int(match) for match in doc_matches]
                logger.info(f"📊 LÍMITES USADOS: {limits_used} (mín configurado: {self.config.search_limit})")

            # Asignar los identificadores encontrados
            metadata.law_identifiers_used = list(set(law_ids + law_identifiers))
            metadata.item_ids_used = list(set(item_ids))

            logger.info(
                f"📊 METADATA: {len(metadata.law_identifiers_used)} leyes, {len(metadata.item_ids_used)} artículos, {metadata.documents_found} docs")

        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo metadata detallada: {e}")
            # Metadata básica en caso de error
            pass

        return metadata

    def _extract_dynamic_response(self, result) -> Union[str, BaseModel]:
        """
        Extrae y valida la respuesta del agente.
        Regla estricta:
        - SOLO se usa result.final_output
        - Debe llegar ya como BaseModel (ideal con AgentOutputSchema)
        - Se permite dict -> se valida contra self.response_model
        - No se accede a result.messages
        """
        out = getattr(result, "final_output", None)
        if out is None:
            logger.error("❌ EXTRACCIÓN: No hay final_output")
            raise RuntimeError("El agente no devolvió final_output (posible fallo de esquema u orquestación)")

        if isinstance(out, BaseModel):
            logger.info(f"✅ EXTRACCIÓN: BaseModel válido ({type(out).__name__})")
            return out

        if isinstance(out, dict):
            try:
                validated = self.response_model.model_validate(out)
                logger.info(f"✅ EXTRACCIÓN: Dict validado contra {self.response_model.__name__}")
                return validated
            except Exception as e:
                logger.error(f"❌ EXTRACCIÓN: Dict no válido contra {self.response_model.__name__}: {e}")
                raise RuntimeError(f"final_output no valida contra {self.response_model.__name__}: {e}") from e

        if self.config.output_format not in ("structured", "model"):
            logger.warning(f"⚠️ EXTRACCIÓN: Convirtiendo {type(out)} a string")
            return str(out)

        logger.error(f"❌ EXTRACCIÓN: Tipo inesperado {type(out)}")
        raise RuntimeError(f"Se esperaba {self.response_model.__name__} o dict; llegó {type(out)}")

    def _analyze_agent_result(self, result) -> None:
        """
        Analiza el resultado del agente para métricas internas.
        Método privado para tracking interno.
        """
        # Analizar el resultado
        if hasattr(result, 'last_agent') and result.last_agent:
            agent_name = getattr(result.last_agent, 'name', 'Desconocido')
        else:
            agent_name = 'Desconocido'
        logger.info(f"🎭 AGENTE FINAL: {agent_name}")

        # Métricas internas
        if hasattr(result, 'messages') and len(result.messages) > 2:
            self._handoffs_simulated += 1
            logger.info(f"🔄 HANDOFF detectado (mensajes: {len(result.messages)})")

        # Trazas de herramientas (simple)
        result_str = str(result)
        if "search_legal_documents" in result_str:
            self._tools_used.append("search_legal_documents")
            logger.info("🔧 Herramienta usada: search_legal_documents")
        if "search_legal_documents_hybrid" in result_str:
            self._tools_used.append("search_legal_documents_hybrid")
            logger.info("🔧 Herramienta usada: search_legal_documents_hybrid")

    def _safe_build_response(self, message: str, error: Optional[Exception] = None) -> BaseModel:
        """
        Construye una instancia válida del modelo dinámico (self.response_model)
        rellenando campos requeridos con valores dummy si es necesario.
        """
        fields = self.response_model.model_fields
        data = {}

        for name, field in fields.items():
            ann = field.annotation
            # Si el campo es str
            if ann == str:
                if "question" in name.lower():
                    data[name] = f"Error: {message}"
                elif "tip" in name.lower():
                    data[name] = f"Detalle: {str(error) if error else message}"
                else:
                    data[name] = message
            # Si es int
            elif ann == int:
                data[name] = 1
            # Si es float
            elif ann == float:
                data[name] = 0.0
            # Si es bool
            elif ann == bool:
                data[name] = False
            # Si es lista
            elif ann and hasattr(ann, "__origin__") and ann.__origin__ == list:
                data[name] = []
            else:
                data[name] = None  # fallback genérico

        return self.response_model.model_validate(data)

    def _handle_timeout_error(self) -> Tuple[Union[str, BaseModel], QueryMetadata]:
        logger.error(f"⏰ TIMEOUT: Consulta tardó más de {self.config.agent_timeout}s")

        metadata = QueryMetadata(
            processing_time_seconds=self.config.agent_timeout,
            agent_calls=self._agent_calls,
            tools_used=["timeout_error"]
        )

        if self.config.output_format in ("structured", "model"):
            response = self._safe_build_response("La consulta tardó demasiado tiempo.", None)
            return response, metadata
        return "La consulta tardó demasiado tiempo.", metadata

    def _handle_general_error(self, error: Exception) -> Tuple[Union[str, BaseModel], QueryMetadata]:
        logger.error(f"❌ ERROR EN CONSULTA: {str(error)}")

        metadata = QueryMetadata(
            processing_time_seconds=0.0,
            agent_calls=self._agent_calls,
            tools_used=["error"]
        )

        if self.config.output_format in ("structured", "model"):
            response = self._safe_build_response("Error interno del sistema", error)
            return response, metadata
        return f"Error interno del sistema: {str(error)}", metadata

    # ===== MÉTODOS PÚBLICOS =====

    async def process_query(self, query: str) -> Tuple[Union[str, BaseModel], QueryMetadata]:
        """Ejecuta el coordinador y devuelve el modelo tipado/texto + metadata completa."""
        start_time = time.time()
        self._agent_calls += 1

        logger.info("=" * 70)
        logger.info(f"🎯 CONSULTA #{self._agent_calls}: {query[:100]}...")
        logger.info(f"📏 LÍMITE MÍNIMO CONFIGURADO: {self.config.search_limit}")
        logger.info("=" * 70)

        try:
            # Ejecuta el flujo con timeout
            logger.info(f"🏃 Ejecutando coordinador (timeout: {self.config.agent_timeout}s)...")
            result = await asyncio.wait_for(
                Runner.run(self._coordinator, query),
                timeout=self.config.agent_timeout
            )

            # Analizar el resultado (métricas internas)
            self._analyze_agent_result(result)

            # Extrae de final_output (nunca de messages)
            logger.info("📤 Extrayendo respuesta final...")
            response = self._extract_dynamic_response(result)

            processing_time = time.time() - start_time

            # Generar metadata completa
            metadata = self._extract_metadata_from_result(result, processing_time)

            logger.info(f"✅ CONSULTA COMPLETADA en {processing_time:.2f}s")
            logger.info(f"📊 METADATA: {metadata.documents_found} docs, {len(metadata.law_identifiers_used)} leyes")
            logger.info("=" * 70)

            return response, metadata

        except asyncio.TimeoutError:
            return self._handle_timeout_error()
        except Exception as e:
            return self._handle_general_error(e)

    def update_response_model(self, new_model: Type[BaseModel], output_format: str = "structured") -> bool:
        """Cambia el modelo de respuesta dinámicamente."""
        try:
            logger.info(f"🔄 Cambiando modelo de respuesta: {self.response_model.__name__} → {new_model.__name__}")
            self.response_model = new_model
            self.config.response_model = new_model
            self.config.output_format = output_format
            self._setup_agents()
            logger.info("✅ Modelo de respuesta actualizado")
            return True
        except Exception as e:
            logger.error(f"❌ Error actualizando modelo de respuesta: {e}")
            return False

    def update_search_limit(self, new_min_limit: int) -> bool:
        """
        Actualiza el límite mínimo de búsqueda y reconfigura los agentes.

        Args:
            new_min_limit: Nuevo límite mínimo (los agentes podrán usar límites más altos)

        Returns:
            bool: True si se actualizó correctamente
        """
        try:
            old_limit = self.config.search_limit
            logger.info(f"🔄 Cambiando límite mínimo de búsqueda: {old_limit} → {new_min_limit}")

            self.config.search_limit = new_min_limit
            global _MIN_SEARCH_LIMIT
            _MIN_SEARCH_LIMIT = new_min_limit

            # Reconfigurar agentes con el nuevo límite
            self._setup_agents()

            logger.info(f"✅ Límite mínimo de búsqueda actualizado a {new_min_limit}")
            return True
        except Exception as e:
            logger.error(f"❌ Error actualizando límite de búsqueda: {e}")
            return False

    def get_current_agent_config(self) -> Dict[str, Any]:
        """Estado actual de configuración de agentes (para debugging)."""
        try:
            return {
                "coordinator": {
                    "name": self._coordinator.name,
                    "model": self._coordinator.model,
                    "instructions_length": len(self._coordinator.instructions) if hasattr(self._coordinator,
                                                                                          'instructions') else 0
                },
                "legal_specialist": {
                    "name": self._legal_specialist.name,
                    "model": self._legal_specialist.model,
                    "has_tools": len(getattr(self._legal_specialist, 'tools', [])) > 0,
                    "tools_count": len(getattr(self._legal_specialist, 'tools', []))
                },
                "general_assistant": {
                    "name": self._general_assistant.name,
                    "model": self._general_assistant.model,
                    "instructions_length": len(self._general_assistant.instructions) if hasattr(self._general_assistant,
                                                                                                'instructions') else 0,
                },
                "service_status": {
                    "agent_calls": self._agent_calls,
                    "handoffs_simulated": self._handoffs_simulated,
                    "tools_used_count": len(self._tools_used),
                    "service_type": self._service_type,
                    "output_format": self.config.output_format,
                    "response_model": self.response_model.__name__,
                    "response_model_fields": list(getattr(self.response_model, 'model_fields', {}).keys()),
                    "min_search_limit": self.config.search_limit,
                    "current_global_limit": _MIN_SEARCH_LIMIT
                }
            }
        except Exception as e:
            logger.error(f"Error obteniendo configuración de agentes: {e}")
            return {"error": str(e)}

    async def run_query(self, query: str) -> Tuple[Union[str, BaseModel], QueryMetadata]:
        """Alias para process_query (compatibilidad)."""
        return await self.process_query(query)

    async def search_documents_directly(self, query: str, limit: Optional[int] = None, ) -> List[Dict[str, Any]]:
        """
        Búsqueda directa con filtro de similitud; no devuelve objetos Pydantic.
        Respeta el límite mínimo configurado si no se especifica uno mayor.
        """
        try:
            effective_limit = limit if limit is not None else self.config.search_limit
            effective_limit = max(effective_limit, self.config.search_limit)

            self._tools_used.append("direct_search")
            logger.info(f"🔍 BÚSQUEDA DIRECTA: consulta='{query}', límite={effective_limit}")

            query_embedding = await self._embedding_service.generate_embedding(query)
            search_results = await self._vector_search.search_similar_vectors(
                embedding=query_embedding,
                limit=effective_limit,
                similarity_threshold=0.3
            )

            results: List[Dict[str, Any]] = []
            for idx, item in enumerate(search_results):
                if item.similarity >= 0.6:
                    results.append({
                        'document_id': str(item.item_id),
                        'content': item.content or "",
                        'similarity': float(item.similarity),
                        'rank': idx + 1,
                        'law_title': item.law_title or "",
                        'law_identifier': item.law_identifier or "",
                        'url': item.url or "",
                        'chapter_info': item.chapter_info or {}
                    })

            logger.info(f"✅ BÚSQUEDA DIRECTA: {len(results)} documentos encontrados (similitud >= 0.6)")
            return results

        except Exception as e:
            logger.exception(f"Error en búsqueda directa: {e}")
            return []

    def get_search_limits_info(self) -> Dict[str, Any]:
        """
        Información sobre los límites de búsqueda configurados y recomendaciones.

        Returns:
            Dict con información sobre límites mínimo, máximo recomendado y guías de uso
        """
        return {
            "min_search_limit": self.config.search_limit,
            "current_global_limit": _MIN_SEARCH_LIMIT,
            "recommended_limits": {
                "simple_queries": f"{self.config.search_limit}-15",
                "complex_queries": "15-30",
                "exhaustive_research": "30-40"
            },
            "usage_guidelines": {
                "simple": "Consultas específicas sobre un artículo o concepto concreto",
                "complex": "Análisis comparativos, múltiples conceptos o investigación detallada",
                "exhaustive": "Investigaciones completas, reports extensos o análisis profundos"
            },
            "agent_autonomy": "Los agentes pueden elegir límites superiores al mínimo según la complejidad de la consulta"
        }