
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel,ConfigDict, Field


class LawItemWithLaw(BaseModel):
    """
    Modelo fusionado que combina información de law_items y laws
    para los resultados de búsqueda
    """

    model_config = ConfigDict(
        from_attributes=True,
        extra='ignore',  # Ignorar campos extra para evitar errores
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None
        }
    )

    # Campos de law_items
    item_id: int
    law_id: str
    second_id: str
    item_title: Optional[str] = None
    content: Optional[str] = None
    url: Optional[str] = None
    num_questions: Optional[int] = None
    max_number_questions: Optional[int] = None
    chapter_info: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    item_updated_at: Optional[datetime] = None

    # Campos de laws
    law_unic_id: Optional[int] = None
    law_identifier: str
    law_title: Optional[str] = None
    last_update: Optional[datetime] = None
    diario: Optional[str] = None
    scope: Optional[Dict[str, Any]] = None
    department: Optional[Dict[str, Any]] = None
    rank: Optional[Dict[str, Any]] = None
    disposition_date: Optional[datetime] = None
    publication_date: Optional[datetime] = None
    validity_date: Optional[datetime] = None
    expired_validity: Optional[bool] = None
    url_eli: Optional[str] = None
    url_html_consolidated: Optional[str] = None

    # Campo calculado de similitud
    similarity: float = 0.0


class LawItemSimplify(BaseModel):
    law_unic_id: Optional[int] = Field(None, description="ID único de la ley")
    law_item_id: int = Field(description="ID único del artículo específico")
    law_title: Optional[str] = Field(None, description="Título de la ley")
    law_identifier: Optional[str] = Field(None, description="Identificador oficial (BOE)")
    item_title: Optional[str] = Field(None, description="Título del artículo")
    content: Optional[str] = Field(None, description="Contenido del artículo")
    url: Optional[str] = Field(None, description="URL oficial")
    model_config = ConfigDict(from_attributes=True, extra="forbid")