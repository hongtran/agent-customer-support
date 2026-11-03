import json

import httpx
import pytest
import respx

from agent_customer_support.llm.schemas import BugReport
from agent_customer_support.mantis import MantisClient, MantisFile
from agent_customer_support.models import Requester

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


async def _create(client: MantisClient, files=None, handler_name=None):
    return await client.create_issue(
        handler_name=handler_name,
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


@respx.mock
async def test_handler_is_sent_only_when_set():
    route = respx.post(f"{BASE}/api/rest/issues").mock(
        return_value=httpx.Response(201, json={"issue": {"id": 7}})
    )
    await _create(_client())
    assert "handler" not in json.loads(route.calls[0].request.content)
    await _create(_client(), handler_name="nguyenvana")
    assert json.loads(route.calls[1].request.content)["handler"] == {"name": "nguyenvana"}


@respx.mock
async def test_rejected_handler_files_the_ticket_unassigned():
    route = respx.post(f"{BASE}/api/rest/issues").mock(
        side_effect=[
            httpx.Response(400, json={"message": "User not found"}),
            httpx.Response(201, json={"issue": {"id": 8}}),
        ]
    )
    issue = await _create(_client(), handler_name="no-such-user")
    assert issue is not None and issue.id == 8
    assert len(route.calls) == 2
    assert "handler" not in json.loads(route.calls[1].request.content)


@respx.mock
async def test_server_error_with_handler_is_not_retried():
    route = respx.post(f"{BASE}/api/rest/issues").mock(return_value=httpx.Response(500))
    assert await _create(_client(), handler_name="nguyenvana") is None
    assert len(route.calls) == 1


# ---- feature requests ----


async def _create_feature(client: MantisClient, handler_name=None, files=None):
    return await client.create_feature_request(
        title="Thêm cột ngày lấy mẫu",
        summary="Muốn xuất thêm cột ngày lấy mẫu trong báo cáo.",
        requester=Requester(name="Nguyễn Văn A", position="Trưởng phòng", department="QA/QC"),
        customer_id="c1",
        customer_name="Công ty ABC",
        application="Lấy mẫu - Quan trắc",
        reason="Sở yêu cầu",
        files=files,
        handler_name=handler_name,
    )


@respx.mock
async def test_feature_request_posts_feature_issue():
    route = respx.post(f"{BASE}/api/rest/issues").mock(
        return_value=httpx.Response(201, json={"issue": {"id": 9}})
    )
    issue = await _create_feature(_client())
    assert issue is not None and issue.id == 9 and issue.url.endswith("view.php?id=9")
    body = json.loads(route.calls[0].request.content)
    assert body["summary"] == "[FEATURE][Công ty ABC] Thêm cột ngày lấy mẫu"
    assert body["description"] == "Muốn xuất thêm cột ngày lấy mẫu trong báo cáo."
    assert body["severity"] == {"name": "feature"}
    info = body["additional_information"]
    assert info.startswith("Người yêu cầu:")
    assert "- Tên nhân sự: Nguyễn Văn A" in info
    assert "- Chức vụ: Trưởng phòng" in info
    assert "- Phòng ban: QA/QC" in info
    assert "Lấy mẫu - Quan trắc" in info and "Lý do: Sở yêu cầu" in info
    assert "handler" not in body
    # no files: no note call
    assert len(respx.calls) == 1


@respx.mock
async def test_feature_request_files_go_in_a_note():
    respx.post(f"{BASE}/api/rest/issues").mock(
        return_value=httpx.Response(201, json={"issue": {"id": 9}})
    )
    note = respx.post(f"{BASE}/api/rest/issues/9/notes").mock(return_value=httpx.Response(201))
    files = [
        MantisFile(name="a.xlsx", content_b64="aGVsbG8="),
        MantisFile(name="b.png", content_b64="eA=="),
    ]
    issue = await _create_feature(_client(), files=files)
    assert issue is not None and issue.id == 9
    body = json.loads(note.calls[0].request.content)
    assert body["files"] == [
        {"name": "a.xlsx", "content": "aGVsbG8="},
        {"name": "b.png", "content": "eA=="},
    ]
    assert "a.xlsx, b.png" in body["text"]


@respx.mock
async def test_failed_file_note_still_returns_the_issue():
    respx.post(f"{BASE}/api/rest/issues").mock(
        return_value=httpx.Response(201, json={"issue": {"id": 9}})
    )
    respx.post(f"{BASE}/api/rest/issues/9/notes").mock(return_value=httpx.Response(413))
    files = [MantisFile(name="a.xlsx", content_b64="aGVsbG8=")]
    issue = await _create_feature(_client(), files=files)
    assert issue is not None and issue.id == 9


@respx.mock
async def test_feature_request_rejected_handler_files_unassigned():
    route = respx.post(f"{BASE}/api/rest/issues").mock(
        side_effect=[
            httpx.Response(400, json={"message": "User not found"}),
            httpx.Response(201, json={"issue": {"id": 10}}),
        ]
    )
    issue = await _create_feature(_client(), handler_name="no-such-user")
    assert issue is not None and issue.id == 10
    assert json.loads(route.calls[0].request.content)["handler"] == {"name": "no-such-user"}
    assert "handler" not in json.loads(route.calls[1].request.content)


@respx.mock
async def test_feature_request_failure_returns_none():
    respx.post(f"{BASE}/api/rest/issues").mock(return_value=httpx.Response(500))
    assert await _create_feature(_client()) is None


@respx.mock
async def test_feature_request_disabled_makes_no_call():
    route = respx.post(f"{BASE}/api/rest/issues").mock(return_value=httpx.Response(201))
    assert await _create_feature(_client(api_token="")) is None
    assert not route.called
