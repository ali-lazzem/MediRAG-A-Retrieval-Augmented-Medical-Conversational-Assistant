from rest_framework import serializers

class AskRequestSerializer(serializers.Serializer):
    question = serializers.CharField(min_length=3)
    top_k = serializers.IntegerField(min_value=1, max_value=20, default=5)
    session_id = serializers.CharField(required=False, allow_null=True)

class ChunkInfoSerializer(serializers.Serializer):
    rank = serializers.IntegerField()
    score = serializers.FloatField()
    source_name = serializers.CharField()
    source = serializers.CharField()           # replaces page
    section = serializers.CharField()
    content = serializers.CharField()

class AskResponseSerializer(serializers.Serializer):
    question = serializers.CharField()
    answer = serializers.CharField()
    chunks_used = ChunkInfoSerializer(many=True)
    retrieval_ms = serializers.FloatField()
    generation_ms = serializers.FloatField()
    total_ms = serializers.FloatField()
    session_id = serializers.CharField()

class SessionInfoSerializer(serializers.Serializer):
    session_id = serializers.CharField()
    title = serializers.CharField()
    timestamp = serializers.FloatField()
    message_count = serializers.IntegerField()