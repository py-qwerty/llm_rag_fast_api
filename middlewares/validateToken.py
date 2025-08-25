import jwt
from jwt import PyJWTError
import os
from dotenv import load_dotenv, find_dotenv
from fastapi import Request, HTTPException

# 🔹 Cargar variables de entorno
load_dotenv(find_dotenv())
JWT_SIGNATURE = os.getenv("JWT_SIGNATURE")

def validate_token(token: str):
    try:
        payload = jwt.decode(
            token,
            JWT_SIGNATURE,
            algorithms=["HS256"],
            audience="authenticated",  # 👈 el aud del token, normalmente "authenticated"
            options={"verify_exp": True}  # verifica expiración
        )
        return payload

    except jwt.ExpiredSignatureError:
        return {"error": "Token has expired"}
    except PyJWTError:
        return {"error": "Invalid token"}


def auth_dependency(request: Request):
    """
    Dependency para validar JWT.
    Devuelve el payload si el token es válido.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ")[1]
    payload = validate_token(token)

    if "error" in payload:
        raise HTTPException(status_code=401, detail=payload["error"])

    return payload
