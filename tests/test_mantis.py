import json

import httpx
import pytest
import respx

from agent_customer_support.llm.schemas import BugReport
from agent_customer_support.mantis import MantisClient, MantisFile

pytestmark = pytest.mark.asyncio

BASE = "https://mantis.example"


def _client(**kw) -> MantisClient:
    return MantisClient(
        base_url=kw.pop("base_url", BASE),
        api_token=kw.pop("api_token", "tok-123"),
        project="CenLab",
        category="General",
        **kw,
    )


def _report() -> BugReport:
    return BugReport(
        title="Import ký hiệu mẫu báo lỗi 500",
        summary="Người dùng import file mẫu thì hệ thống báo lỗi 500.",
        steps_to_reproduce="1. Vào Lấy mẫu\n2. Bấm Import",
    )


async def _create(client: MantisClient, files=None):
    return await client.create_issue(
        report=_report(),
        customer_id="c1",
        customer_name="Công ty ABC",
        application="Lấy mẫu - Quan trắc",
        transcript="user: bị lỗi\nassistant: gửi ảnh giúp",
        files=files or [],
    )


@respx.mock
async def test_create_issue_posts_report_and_returns_issue():
    route = respx.post(f"{BASE}/api/rest/issues").mock(
        return_value=httpx.Response(201, json={"issue": {"id": 7}})
    )
    issue = await _create(_client())
    assert issue is not None
    assert issue.id == 7
    assert issue.url == f"{BASE}/view.php?id=7"
    req = route.calls[0].request
    assert req.headers["Authorization"] == "tok-123"
    body = json.loads(req.content)
    assert body["summary"] == "[BUG][Công ty ABC] Import ký hiệu mẫu báo lỗi 500"
    assert body["description"] == "Người dùng import file mẫu thì hệ thống báo lỗi 500."
    assert body["steps_to_reproduce"] == "1. Vào Lấy mẫu\n2. Bấm Import"
    assert body["project"] == {"name": "CenLab"}
    assert body["category"] == {"name": "General"}
    assert "c1" in body["additional_information"]
    assert "Lấy mẫu - Quan trắc" in body["additional_information"]
    assert "user: bị lỗi" in body["additional_information"]


@respx.mock
async def test_create_issue_attaches_files_in_second_call():
    respx.post(f"{BASE}/api/rest/issues").mock(
        return_value=httpx.Response(201, json={"issue": {"id": 7}})
    )
    files_route = respx.post(f"{BASE}/api/rest/issues/7/files").mock(
        return_value=httpx.Response(201)
    )
    issue = await _create(
        _client(), files=[MantisFile(name="screenshot-1.png", content_b64="aGVsbG8=")]
    )
    assert issue is not None and issue.id == 7
    body = json.loads(files_route.calls[0].request.content)
    assert body == {"files": [{"name": "screenshot-1.png", "content": "aGVsbG8="}]}


@respx.mock
async def test_file_upload_failure_keeps_issue():
    respx.post(f"{BASE}/api/rest/issues").mock(
        return_value=httpx.Response(201, json={"issue": {"id": 7}})
    )
    respx.post(f"{BASE}/api/rest/issues/7/files").mock(return_value=httpx.Response(400))
    issue = await _create(_client(), files=[MantisFile(name="a.png", content_b64="aGVsbG8=")])
    assert issue is not None and issue.id == 7


@respx.mock
async def test_create_failure_returns_none_without_raising():
    respx.post(f"{BASE}/api/rest/issues").mock(return_value=httpx.Response(500))
    assert await _create(_client()) is None


@respx.mock
async def test_network_error_returns_none_without_raising():
    respx.post(f"{BASE}/api/rest/issues").mock(side_effect=httpx.ConnectError("down"))
    assert await _create(_client()) is None


@respx.mock
async def test_disabled_when_token_missing_makes_no_call():
    route = respx.post(f"{BASE}/api/rest/issues").mock(
        return_value=httpx.Response(201, json={"issue": {"id": 7}})
    )
    # None means "read Settings" (same rule as Escalator); "" is the explicit off switch,
    # so this test holds even when the developer's .env carries real credentials.
    client = _client(api_token="")
    assert client.enabled is False
    assert await _create(client) is None
    assert not route.called


@respx.mock
async def test_title_is_truncated_to_mantis_limit():
    route = respx.post(f"{BASE}/api/rest/issues").mock(
        return_value=httpx.Response(201, json={"issue": {"id": 1}})
    )
    long_report = BugReport(title="x" * 200, summary="s", steps_to_reproduce="")
    await _client().create_issue(
        report=long_report,
        customer_id="c1",
        customer_name="Công ty ABC",
        application=None,
        transcript="",
        files=[],
    )
    body = json.loads(route.calls[0].request.content)
    assert len(body["summary"]) == 128
    # the prefix survives; only the title is cut
    assert body["summary"].startswith("[BUG][Công ty ABC] xxx")
    assert body["summary"].endswith("x")


async def test_title_falls_back_to_customer_id_when_name_is_blank():
    with respx.mock:
        route = respx.post(f"{BASE}/api/rest/issues").mock(
            return_value=httpx.Response(201, json={"issue": {"id": 1}})
        )
        await _client().create_issue(
            report=_report(),
            customer_id="c1",
            customer_name="",
            application=None,
            transcript="",
            files=[],
        )
    body = json.loads(route.calls[0].request.content)
    assert body["summary"].startswith("[BUG][c1] ")


async def test_trailing_slash_in_base_url_is_tolerated():
    client = _client(base_url=f"{BASE}/")
    with respx.mock:
        route = respx.post(f"{BASE}/api/rest/issues").mock(
            return_value=httpx.Response(201, json={"issue": {"id": 3}})
        )
        issue = await _create(client)
    assert route.called and issue is not None and issue.url == f"{BASE}/view.php?id=3"


@respx.mock
async def test_add_note_posts_text_and_returns_true():
    route = respx.post(f"{BASE}/api/rest/issues/7/notes").mock(
        return_value=httpx.Response(201, json={"note": {"id": 3}})
    )
    assert await _client().add_note(7, "Liên hệ: SĐT 0912345678") is True
    req = route.calls[0].request
    assert req.headers["Authorization"] == "tok-123"
    assert json.loads(req.content) == {"text": "Liên hệ: SĐT 0912345678"}


@respx.mock
async def test_add_note_failure_returns_false_without_raising():
    respx.post(f"{BASE}/api/rest/issues/7/notes").mock(return_value=httpx.Response(404))
    assert await _client().add_note(7, "x") is False
    respx.post(f"{BASE}/api/rest/issues/8/notes").mock(side_effect=httpx.ConnectError("down"))
    assert await _client().add_note(8, "x") is False


@respx.mock
async def test_add_note_disabled_makes_no_call():
    route = respx.post(f"{BASE}/api/rest/issues/7/notes").mock(return_value=httpx.Response(201))
    assert await _client(api_token="").add_note(7, "x") is False
    assert not route.called
