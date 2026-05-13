# FILE: rag/admin.py

from django.contrib import admin
from .models import Conversation, Message, Profile

admin.site.register(Conversation)
admin.site.register(Message)
admin.site.register(Profile)