# FILE: rag/views.py

import time
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Conversation, Message, Profile
from .serializers import (
    AskRequestSerializer, AskResponseSerializer,
    SessionInfoSerializer
)
from .retriever import retrieve
from .llm import answer
from .forms import CustomUserCreationForm

# ----- Landing Page -----
def home(request):
    return render(request, 'home.html')

# ----- Authentication Views -----
def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('chat')
        else:
            return render(request, 'registration/login.html', {'error': 'Invalid credentials'})
    return render(request, 'registration/login.html')

def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('chat')
        else:
            return render(request, 'registration/register.html', {'form': form})
    else:
        form = CustomUserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('home')

# ----- Chat interface (requires login) -----
@login_required
def index(request):
    return render(request, 'index.html')

# ----- API Endpoints (all require login) -----
@api_view(['POST'])
@login_required
def ask(request):
    serializer = AskRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    question = serializer.validated_data['question']
    top_k = serializer.validated_data['top_k']
    session_id = serializer.validated_data.get('session_id')

    if not session_id:
        conversation = Conversation.objects.create(
            user=request.user,
            title=question[:50]
        )
        session_id = str(conversation.id)
    else:
        try:
            conversation = Conversation.objects.get(id=int(session_id), user=request.user)
        except (Conversation.DoesNotExist, ValueError):
            conversation = Conversation.objects.create(
                user=request.user,
                title=question[:50]
            )
            session_id = str(conversation.id)

    # Check if this is the first message in the conversation
    is_first_message = conversation.messages.count() == 0

    # Save user message
    Message.objects.create(
        conversation=conversation,
        sender='user',
        content=question
    )

    # If first message, update conversation title to the question (trimmed)
    if is_first_message:
        new_title = question[:60] + ('…' if len(question) > 60 else '')
        conversation.title = new_title
        conversation.save(update_fields=['title'])

    t0 = time.perf_counter()
    chunks = retrieve(question, top_k=top_k)
    t1 = time.perf_counter()

    if not chunks:
        error_msg = "I could not find relevant information to answer your question."
        Message.objects.create(
            conversation=conversation,
            sender='bot',
            content=error_msg
        )
        return Response({
            "error": "No relevant chunks found.",
            "session_id": session_id
        }, status=status.HTTP_404_NOT_FOUND)

    # Get recent history (last 5 messages) for context
    history_messages = conversation.messages.order_by('-timestamp')[:5][::-1]
    history = [{"role": m.sender, "content": m.content} for m in history_messages if m.sender in ('user', 'bot')]
    for h in history:
        if h['role'] == 'bot':
            h['role'] = 'assistant'

    generated_answer = answer(chunks, question, history)
    t2 = time.perf_counter()

    Message.objects.create(
        conversation=conversation,
        sender='bot',
        content=generated_answer
    )

    retrieval_ms = round((t1 - t0) * 1000, 2)
    generation_ms = round((t2 - t1) * 1000, 2)
    total_ms = round((t2 - t0) * 1000, 2)

    chunks_info = [
        {
            "rank": i + 1,
            "score": round(c.get("score", 0.0), 4),
            "source_name": c.get("source_name", "Unknown"),
            "source": str(c.get("source", "CSV dataset")),
            "section": c.get("section", "General"),
            "content": c.get("content", ""),
        }
        for i, c in enumerate(chunks)
    ]

    response_data = {
        "question": question,
        "answer": generated_answer,
        "chunks_used": chunks_info,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "total_ms": total_ms,
        "session_id": str(conversation.id),
    }
    return Response(response_data, status=status.HTTP_200_OK)

@api_view(['GET'])
@login_required
def list_sessions(request):
    conversations = Conversation.objects.filter(user=request.user).order_by('-created_at')
    sessions = []
    for conv in conversations:
        sessions.append({
            "session_id": str(conv.id),
            "title": conv.title or "Untitled",
            "timestamp": conv.created_at.timestamp(),
            "message_count": conv.messages.count()
        })
    serializer = SessionInfoSerializer(sessions, many=True)
    return Response(serializer.data)

@api_view(['POST'])
@login_required
def create_session(request):
    title = request.data.get('title', 'New Conversation')
    conversation = Conversation.objects.create(user=request.user, title=title)
    return Response({"session_id": str(conversation.id)})

@api_view(['PUT'])
@login_required
def rename_session(request, session_id):
    try:
        conversation = Conversation.objects.get(id=int(session_id), user=request.user)
        new_title = request.data.get('title', '').strip()
        if not new_title:
            return Response({"error": "Title cannot be empty"}, status=status.HTTP_400_BAD_REQUEST)
        conversation.title = new_title
        conversation.save()
        return Response({"session_id": str(conversation.id), "title": conversation.title})
    except (Conversation.DoesNotExist, ValueError):
        return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)

@api_view(['DELETE'])
@login_required
def delete_session(request, session_id):
    try:
        conversation = Conversation.objects.get(id=int(session_id), user=request.user)
        conversation.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    except (Conversation.DoesNotExist, ValueError):
        return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)

@api_view(['GET'])
@login_required
def get_session_messages(request, session_id):
    try:
        conversation = Conversation.objects.get(id=int(session_id), user=request.user)
        messages = conversation.messages.all()
        data = [
            {
                "role": m.sender,
                "content": m.content,
                "timestamp": m.timestamp.timestamp()
            }
            for m in messages
        ]
        return Response(data)
    except (Conversation.DoesNotExist, ValueError):
        return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)

@api_view(['GET'])
def health(request):
    return Response({"status": "ok"})