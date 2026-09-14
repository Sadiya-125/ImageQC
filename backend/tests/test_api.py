"""
Integration tests against the real FastAPI app, exercised over HTTP (via
httpx's ASGI transport) with a real Postgres test database -- see
conftest.py for the `client`/`db_session` fixtures and why a real Postgres
is used instead of SQLite.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CLEAN_IMAGE = REPO_ROOT / "sample_images" / "clean" / "I15.png"
SAMPLE_CORRUPTED_IMAGE = REPO_ROOT / "sample_images" / "corrupted" / "I15_10_05.png"


async def _upload(client, path: Path, filename: str, content_type: str = "image/png"):
    with open(path, "rb") as f:
        return await client.post(
            "/api/analyze", files={"file": (filename, f.read(), content_type)}
        )


@pytest.mark.asyncio
class TestHealth:
    async def test_health_ok(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


@pytest.mark.asyncio
class TestAnalyzeEndpoint:
    async def test_analyze_real_sample_image(self, client):
        response = await _upload(client, SAMPLE_CLEAN_IMAGE, "clean.png")
        assert response.status_code == 201

        body = response.json()
        assert body["filename"] == "clean.png"
        assert 0.0 <= body["quality_score"] <= 100.0
        assert body["quality_label"] in {"ACCEPTABLE", "DEGRADED", "DEFECTIVE"}
        assert isinstance(body["issues"], list)
        assert "laplacian_variance" in body["image_stats"]
        assert body["gradcam_available"] is True

    async def test_analyze_deliberately_corrupt_file_returns_400(self, client):
        response = await client.post(
            "/api/analyze",
            files={"file": ("not_an_image.png", b"this is definitely not a png", "image/png")},
        )
        assert response.status_code == 400
        assert "not a readable image" in response.json()["detail"].lower()

    async def test_analyze_empty_file_returns_400(self, client):
        response = await client.post(
            "/api/analyze", files={"file": ("empty.png", b"", "image/png")}
        )
        assert response.status_code == 400

    async def test_analyze_persists_to_db(self, client):
        response = await _upload(client, SAMPLE_CORRUPTED_IMAGE, "corrupted.png")
        assert response.status_code == 201
        analysis_id = response.json()["id"]

        detail_response = await client.get(f"/api/analyses/{analysis_id}")
        assert detail_response.status_code == 200
        assert detail_response.json()["filename"] == "corrupted.png"


@pytest.mark.asyncio
class TestAnalysesListEndpoint:
    async def test_pagination(self, client):
        for i in range(3):
            resp = await _upload(client, SAMPLE_CLEAN_IMAGE, f"clean-{i}.png")
            assert resp.status_code == 201

        page1 = await client.get("/api/analyses", params={"page": 1, "page_size": 2})
        assert page1.status_code == 200
        body1 = page1.json()
        assert body1["total"] >= 3
        assert len(body1["items"]) == 2
        assert body1["page"] == 1
        assert body1["page_size"] == 2

        page2 = await client.get("/api/analyses", params={"page": 2, "page_size": 2})
        body2 = page2.json()
        assert len(body2["items"]) >= 1
        # No overlap between pages.
        ids_page1 = {item["id"] for item in body1["items"]}
        ids_page2 = {item["id"] for item in body2["items"]}
        assert ids_page1.isdisjoint(ids_page2)

    async def test_list_excludes_heavy_fields(self, client):
        await _upload(client, SAMPLE_CLEAN_IMAGE, "clean.png")
        response = await client.get("/api/analyses")
        item = response.json()["items"][0]
        assert "image_data" not in item
        assert "image_stats" not in item

    async def test_get_missing_analysis_returns_404(self, client):
        response = await client.get("/api/analyses/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404

    async def test_gradcam_endpoint_returns_png(self, client):
        upload_response = await _upload(client, SAMPLE_CORRUPTED_IMAGE, "corrupted.png")
        analysis_id = upload_response.json()["id"]

        gradcam_response = await client.get(
            f"/api/analyses/{analysis_id}/gradcam", params={"head": "corruption"}
        )
        assert gradcam_response.status_code == 200
        assert gradcam_response.headers["content-type"] == "image/png"
        assert len(gradcam_response.content) > 0

    async def test_gradcam_rejects_unknown_head(self, client):
        upload_response = await _upload(client, SAMPLE_CLEAN_IMAGE, "clean.png")
        analysis_id = upload_response.json()["id"]

        response = await client.get(
            f"/api/analyses/{analysis_id}/gradcam", params={"head": "not_a_real_head"}
        )
        assert response.status_code == 400

    async def test_image_endpoint_returns_original_bytes(self, client):
        upload_response = await _upload(client, SAMPLE_CLEAN_IMAGE, "clean.png")
        analysis_id = upload_response.json()["id"]

        image_response = await client.get(f"/api/analyses/{analysis_id}/image")
        assert image_response.status_code == 200
        assert image_response.content == SAMPLE_CLEAN_IMAGE.read_bytes()
