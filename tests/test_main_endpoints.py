"""Tests for main.py endpoints.

Uses FastAPI TestClient for HTTP-level testing.
Tests health endpoints, settings, conversations, models, and glossary.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory for tests."""
    data_dir = tmp_path / "data"
    conversations_dir = data_dir / "conversations"
    conversations_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def mock_storage(temp_data_dir, monkeypatch):
    """Mock storage module to use temp directory."""
    monkeypatch.setattr("backend.storage.DATA_DIR", str(temp_data_dir / "conversations"))
    monkeypatch.setattr("backend.config.DATA_DIR", str(temp_data_dir / "conversations"))
    return temp_data_dir


# =============================================================================
# Health Endpoints
# =============================================================================


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client):
        """Root endpoint returns ok status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "LLM Council Plus"

    def test_health_endpoint(self, client):
        """Health endpoint returns health status and version info."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "LLM Council Plus API"
        assert "version" in data
        assert "api_versions" in data
        assert data["api_versions"]["current"] == "v1"
        assert "v1" in data["api_versions"]["supported"]

    def test_health_includes_deprecated_versions(self, client):
        """Health endpoint lists deprecated API versions."""
        response = client.get("/health")
        data = response.json()
        assert "deprecated" in data["api_versions"]
        # v0 should be listed as deprecated
        deprecated = data["api_versions"]["deprecated"]
        assert any("v0" in d or "/api/claims" in d for d in deprecated)


# =============================================================================
# Settings Endpoints
# =============================================================================


class TestSettingsGet:
    """Tests for GET /api/settings endpoint."""

    def test_get_settings_success(self, client):
        """Get settings returns current configuration."""
        response = client.get("/api/settings")
        assert response.status_code == 200
        data = response.json()
        # Check expected keys exist
        assert "search_provider" in data
        assert "ollama_base_url" in data
        assert "council_models" in data
        assert "chairman_model" in data

    def test_get_settings_hides_api_keys(self, client):
        """API keys are not returned, only their set status."""
        response = client.get("/api/settings")
        data = response.json()
        # Should have _set flags but not actual keys
        assert "tavily_api_key_set" in data
        assert "openrouter_api_key_set" in data
        assert "tavily_api_key" not in data
        assert "openrouter_api_key" not in data


class TestSettingsDefaults:
    """Tests for GET /api/settings/defaults endpoint."""

    def test_get_defaults_success(self, client):
        """Get default settings returns factory defaults."""
        response = client.get("/api/settings/defaults")
        assert response.status_code == 200
        data = response.json()
        assert "council_models" in data
        assert "chairman_model" in data
        assert "enabled_providers" in data
        assert "stage1_prompt" in data
        assert "stage2_prompt" in data
        assert "stage3_prompt" in data


class TestSettingsUpdate:
    """Tests for PUT /api/settings endpoint."""

    def test_update_search_provider_valid(self, client):
        """Update search provider with valid value."""
        response = client.put(
            "/api/settings",
            json={"search_provider": "duckduckgo"},
        )
        assert response.status_code == 200
        assert response.json()["search_provider"] == "duckduckgo"

    def test_update_search_provider_invalid(self, client):
        """Invalid search provider returns 400."""
        response = client.put(
            "/api/settings",
            json={"search_provider": "invalid_provider"},
        )
        assert response.status_code == 400
        assert "Invalid search provider" in response.json()["detail"]

    def test_update_keyword_extraction_valid(self, client):
        """Update keyword extraction mode."""
        response = client.put(
            "/api/settings",
            json={"search_keyword_extraction": "direct"},
        )
        assert response.status_code == 200

    def test_update_keyword_extraction_invalid(self, client):
        """Invalid keyword extraction mode returns 400."""
        response = client.put(
            "/api/settings",
            json={"search_keyword_extraction": "invalid"},
        )
        assert response.status_code == 400

    def test_update_council_models_minimum(self, client):
        """Council models requires at least 2 models."""
        response = client.put(
            "/api/settings",
            json={"council_models": ["ollama:llama3.2"]},
        )
        assert response.status_code == 400
        assert "at least two" in response.json()["detail"].lower()

    def test_update_council_models_maximum(self, client):
        """Council models maximum is 8."""
        too_many = [f"ollama:model{i}" for i in range(10)]
        response = client.put(
            "/api/settings",
            json={"council_models": too_many},
        )
        assert response.status_code == 400
        assert "8" in response.json()["detail"]


# =============================================================================
# Provider Test Endpoints
# =============================================================================


class TestProviderTests:
    """Tests for provider API key testing endpoints."""

    def test_test_tavily_success(self, client):
        """Test Tavily API with mocked success."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = client.post(
                "/api/settings/test-tavily",
                json={"api_key": "test-key"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_test_tavily_invalid_key(self, client):
        """Test Tavily API with invalid key returns failure."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = client.post(
                "/api/settings/test-tavily",
                json={"api_key": "invalid-key"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "Invalid" in data["message"]

    def test_test_brave_success(self, client):
        """Test Brave API with mocked success."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            response = client.post(
                "/api/settings/test-brave",
                json={"api_key": "test-key"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_test_ollama_success(self, client):
        """Test Ollama connection with mocked success."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            response = client.post(
                "/api/settings/test-ollama",
                json={"base_url": "http://localhost:11434"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_test_ollama_connection_refused(self, client):
        """Test Ollama connection when server is down."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            response = client.post(
                "/api/settings/test-ollama",
                json={"base_url": "http://localhost:11434"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "Could not connect" in data["message"]

    def test_test_openrouter_success(self, client):
        """Test OpenRouter API with mocked success."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            response = client.post(
                "/api/settings/test-openrouter",
                json={"api_key": "test-key"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_test_provider_invalid_provider(self, client):
        """Test provider with invalid provider ID returns 400."""
        response = client.post(
            "/api/settings/test-provider",
            json={"provider_id": "nonexistent", "api_key": "test"},
        )
        assert response.status_code == 400
        assert "Invalid provider" in response.json()["detail"]


# =============================================================================
# Conversations Endpoints
# =============================================================================


class TestConversationsList:
    """Tests for GET /api/conversations endpoint."""

    def test_list_conversations_empty(self, client, mock_storage):
        """Empty store returns empty list."""
        response = client.get("/api/conversations")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_conversations_with_data(self, client, mock_storage):
        """Returns conversations after creation."""
        # Create a conversation first
        create_response = client.post("/api/conversations", json={})
        assert create_response.status_code == 200

        response = client.get("/api/conversations")
        assert response.status_code == 200
        conversations = response.json()
        assert len(conversations) >= 1

    def test_list_conversations_metadata_only(self, client, mock_storage):
        """List returns metadata, not full messages."""
        # Create a conversation
        client.post("/api/conversations", json={})

        response = client.get("/api/conversations")
        conversations = response.json()
        # Should have metadata fields but not full messages array
        if conversations:
            conv = conversations[0]
            assert "id" in conv
            assert "title" in conv
            assert "message_count" in conv


class TestConversationsCreate:
    """Tests for POST /api/conversations endpoint."""

    def test_create_conversation_success(self, client, mock_storage):
        """Create conversation returns new conversation."""
        response = client.post("/api/conversations", json={})
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "created_at" in data
        assert "title" in data
        assert data["messages"] == []

    def test_create_conversation_generates_uuid(self, client, mock_storage):
        """Created conversation has valid UUID id."""
        response = client.post("/api/conversations", json={})
        data = response.json()
        # UUID format check
        assert len(data["id"]) == 36
        assert data["id"].count("-") == 4


class TestConversationsGet:
    """Tests for GET /api/conversations/{id} endpoint."""

    def test_get_conversation_success(self, client, mock_storage):
        """Get existing conversation returns full data."""
        # Create first
        create_response = client.post("/api/conversations", json={})
        conv_id = create_response.json()["id"]

        # Then get
        response = client.get(f"/api/conversations/{conv_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == conv_id
        assert "messages" in data

    def test_get_conversation_not_found(self, client, mock_storage):
        """Non-existent conversation returns 404."""
        response = client.get("/api/conversations/nonexistent-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestConversationsDelete:
    """Tests for DELETE /api/conversations/{id} endpoint."""

    def test_delete_conversation_success(self, client, mock_storage):
        """Delete existing conversation returns success."""
        # Create first
        create_response = client.post("/api/conversations", json={})
        conv_id = create_response.json()["id"]

        # Then delete
        response = client.delete(f"/api/conversations/{conv_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

        # Verify gone
        get_response = client.get(f"/api/conversations/{conv_id}")
        assert get_response.status_code == 404

    def test_delete_conversation_not_found(self, client, mock_storage):
        """Delete non-existent conversation returns 404."""
        response = client.delete("/api/conversations/nonexistent-id")
        assert response.status_code == 404


# =============================================================================
# Model Endpoints
# =============================================================================


class TestModelEndpoints:
    """Tests for model discovery endpoints."""

    def test_get_models_success(self, client):
        """Get models returns model list."""
        with patch("backend.openrouter.fetch_models") as mock_fetch:
            mock_fetch.return_value = [
                {"id": "test:model1", "name": "Test Model 1"},
                {"id": "test:model2", "name": "Test Model 2"},
            ]

            response = client.get("/api/models")
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            # Either returns our mock or the fallback static list
            assert isinstance(data["models"], list)

    def test_get_models_fallback_to_static(self, client):
        """Get models falls back to static list when fetch fails."""
        with patch("backend.openrouter.fetch_models") as mock_fetch:
            mock_fetch.return_value = None  # Simulate failure

            response = client.get("/api/models")
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            # Should have some models from static fallback
            assert isinstance(data["models"], list)

    def test_get_ollama_tags_success(self, client):
        """Get Ollama tags returns available models."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "llama3.2:latest", "modified_at": "2026-01-01"},
                    {"name": "mistral:latest", "modified_at": "2026-01-02"},
                ]
            }
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            response = client.get("/api/ollama/tags")
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert len(data["models"]) == 2
            assert data["models"][0]["is_free"] is True

    def test_get_ollama_tags_connection_error(self, client):
        """Get Ollama tags returns error when server is down."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            response = client.get("/api/ollama/tags")
            assert response.status_code == 200
            data = response.json()
            assert data["models"] == []
            assert "error" in data
            assert "Could not connect" in data["error"]

    def test_get_direct_models(self, client):
        """Get direct models returns models from configured providers."""
        with patch("backend.main.PROVIDERS") as mock_providers:
            # Create mock provider that returns models
            mock_provider = MagicMock()
            mock_provider.get_models = AsyncMock(
                return_value=[{"id": "direct:model1", "name": "Direct Model 1"}]
            )
            mock_providers.items.return_value = [("testprovider", mock_provider)]

            response = client.get("/api/models/direct")
            assert response.status_code == 200
            # Response is a list directly
            data = response.json()
            assert isinstance(data, list)

    def test_get_custom_endpoint_models_not_configured(self, client):
        """Get custom endpoint models returns error when not configured."""
        with patch("backend.settings.get_settings") as mock_settings:
            mock_obj = MagicMock()
            mock_obj.custom_endpoint_url = None
            mock_settings.return_value = mock_obj

            response = client.get("/api/custom-endpoint/models")
            assert response.status_code == 200
            data = response.json()
            assert data["models"] == []
            assert "error" in data


# =============================================================================
# Glossary Endpoints
# =============================================================================


class TestGlossaryEndpoints:
    """Tests for glossary endpoints."""

    def test_get_glossary_success(self, client):
        """Get glossary returns fallacies list."""
        # Create mock glossary file
        glossary_path = Path(__file__).parent.parent / "backend" / "resources" / "fallacies_glossary.json"
        if glossary_path.exists():
            response = client.get("/api/glossary")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
        else:
            # If glossary doesn't exist, should return 404
            response = client.get("/api/glossary")
            assert response.status_code == 404

    def test_get_glossary_not_found(self, client):
        """Get glossary returns 404 when file missing."""
        with patch("pathlib.Path.exists", return_value=False):
            response = client.get("/api/glossary")
            assert response.status_code == 404

    def test_get_fallacy_success(self, client):
        """Get specific fallacy by ID."""
        # Mock the glossary data
        mock_glossary = [
            {"id": "ad_hominem", "name": "Ad Hominem", "description": "Attack the person"},
            {"id": "straw_man", "name": "Straw Man", "description": "Misrepresent argument"},
        ]

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_text", return_value=json.dumps(mock_glossary)):
            response = client.get("/api/glossary/ad_hominem")
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "ad_hominem"

    def test_get_fallacy_not_found(self, client):
        """Get non-existent fallacy returns 404."""
        mock_glossary = [
            {"id": "ad_hominem", "name": "Ad Hominem"},
        ]

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_text", return_value=json.dumps(mock_glossary)):
            response = client.get("/api/glossary/nonexistent")
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()


# =============================================================================
# OpenAI Compatible Endpoints
# =============================================================================


class TestOpenAICompatibleEndpoints:
    """Tests for OpenAI-compatible API endpoints."""

    def test_list_openai_models(self, client):
        """List models endpoint returns roundtable variants."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        # Should include roundtable models
        model_ids = [m["id"] for m in data["data"]]
        assert "roundtable" in model_ids or any("roundtable" in m for m in model_ids)
