import os

def read_root():
    return {
        "message": "The API is running successfully!",
        "status": "healthy",
        "version": "1.0.0",
        "port": os.environ.get("PORT", "unknown"),
        "host": os.environ.get("HOST", "unknown")
    }