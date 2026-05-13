from django.apps import AppConfig

class RagConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'rag'

    def ready(self):
        from . import retriever
        retriever._load()   # loads embedding model, FAISS index, metadata