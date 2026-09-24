import pytest

from agent_customer_support.contact import describe, parse
from agent_customer_support.models import ContactInfo

pytestmark = pytest.mark.asyncio


async def test_mobile_with_spaces():
    c = parse("SĐT của em là 0912 345 678 ạ")
    assert c.phone == "0912345678"
    assert c.email is None
    assert c.found is True


async def test_international_prefix_is_normalised():
    assert parse("+84 912-345-678").phone == "0912345678"
    assert parse("84912345678").phone == "0912345678"


async def test_landline_eleven_digits():
    assert parse("gọi văn phòng 024 3825 1234").phone == "02438251234"


async def test_email_is_lowercased():
    c = parse("mail: Ten.Nguyen@CongTy.vn")
    assert c.email == "ten.nguyen@congty.vn"
    assert c.phone is None
    assert c.found is True


async def test_both_in_one_message():
    c = parse("0912345678, a@b.vn nhé")
    assert (c.phone, c.email) == ("0912345678", "a@b.vn")


async def test_email_digits_are_not_taken_as_phone():
    c = parse("user0912345678@mail.com")
    assert c.email == "user0912345678@mail.com"
    assert c.phone is None


async def test_nothing_found_keeps_raw():
    c = parse("không có")
    assert c.found is False
    assert c.phone is None and c.email is None
    assert c.raw == "không có"


async def test_raw_is_kept_when_found_too():
    assert parse("0912345678 gọi sau 5h").raw == "0912345678 gọi sau 5h"


async def test_describe_formats_both():
    assert describe(ContactInfo(phone="0912345678", email="a@b.vn")) == (
        "SĐT: 0912345678 · Email: a@b.vn"
    )


async def test_describe_one_side_and_none():
    assert describe(ContactInfo(phone="0912345678")) == "SĐT: 0912345678"
    assert describe(ContactInfo(email="a@b.vn")) == "Email: a@b.vn"
    assert describe(ContactInfo()) == "chưa cung cấp"
