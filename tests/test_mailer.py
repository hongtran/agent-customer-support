import json

import httpx
import pytest
import respx

from agent_customer_support.mailer import CSMailer

pytestmark = pytest.mark.asyncio

URL = "https://resend.example/emails"


def _mailer(**kw) -> CSMailer:
    return CSMailer(
        api_key=kw.pop("api_key", "re_test"),
        api_url=URL,
        sender=kw.pop("sender", "bot@example.vn"),
        to=kw.pop("to", "cs1@example.vn, cs2@example.vn"),
        **kw,
    )


@respx.mock
async def test_send_posts_to_resend_with_every_recipient_subject_and_body():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={"id": "e1"}))
    ok = await _mailer().send(subject="[HỖ TRỢ] test", body="nội dung\nhai dòng")
    assert ok is True
    req = route.calls.last.request
    assert req.headers["Authorization"] == "Bearer re_test"
    assert json.loads(req.content) == {
        "from": "bot@example.vn",
        "to": ["cs1@example.vn", "cs2@example.vn"],
        "subject": "[HỖ TRỢ] test",
        "text": "nội dung\nhai dòng",
    }


@respx.mock
async def test_rejected_send_returns_false_without_raising():
    respx.post(URL).mock(
        return_value=httpx.Response(
            403, json={"name": "validation_error", "message": "domain is not verified"}
        )
    )
    assert await _mailer().send(subject="s", body="b") is False


@respx.mock
async def test_network_failure_returns_false_without_raising():
    respx.post(URL).mock(side_effect=httpx.ConnectError("down"))
    assert await _mailer().send(subject="s", body="b") is False


@respx.mock
@pytest.mark.parametrize("missing", ["api_key", "sender", "to"])
async def test_disabled_without_key_sender_or_recipients_makes_no_call(missing):
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={"id": "e1"}))
    m = _mailer(**{missing: ""})
    assert m.enabled is False
    assert await m.send(subject="s", body="b") is False
    assert route.called is False


async def test_recipients_list_is_split_and_trimmed():
    m = _mailer(to=" a@x.vn ,b@x.vn,, ")
    assert m.recipients == ["a@x.vn", "b@x.vn"]


@respx.mock
async def test_cc_list_is_sent_when_set():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={"id": "e1"}))
    m = _mailer(cc=" lead@x.vn ,, boss@x.vn ")
    assert m.cc == ["lead@x.vn", "boss@x.vn"]
    assert await m.send(subject="s", body="b") is True
    assert json.loads(route.calls.last.request.content)["cc"] == [
        "lead@x.vn",
        "boss@x.vn",
    ]


async def test_cc_alone_does_not_enable_the_mailer():
    assert _mailer(to="", cc="lead@x.vn").enabled is False
