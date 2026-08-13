import time

import pytest

from nenapu import connect
from nenapu.models import Skill
from nenapu.skills import UNUSED_AFTER_DAYS, SkillStore


@pytest.fixture
def skills():
    return SkillStore(connect(":memory:"))


def test_upsert_and_search(skills):
    skills.upsert(Skill(name="pdf-extract", body="use pdfplumber", description="extract pdf text"))
    found = skills.search("pdf")
    assert found and found[0].name == "pdf-extract"


def test_repeated_failure_quarantines(skills):
    skills.upsert(Skill(name="flaky", body="..."))
    for _ in range(4):
        skills.record_outcome("flaky", "failure")
    s = skills.get("flaky")
    assert s.status == "quarantined" and "success rate" in s.quarantine_reason


def test_mostly_successful_skill_survives(skills):
    skills.upsert(Skill(name="solid", body="..."))
    for _ in range(4):
        skills.record_outcome("solid", "success")
    skills.record_outcome("solid", "failure")
    assert skills.get("solid").status == "active"


def test_never_used_old_skill_is_quarantined(skills):
    old = time.time() - (UNUSED_AFTER_DAYS + 10) * 86400
    skills.upsert(Skill(name="forgotten", body="...", created_at=old))
    culled = skills.sweep()
    assert [s.name for s in culled] == ["forgotten"]
    assert "never invoked" in skills.get("forgotten").quarantine_reason


def test_quarantined_skills_excluded_from_search(skills):
    skills.upsert(Skill(name="bad", body="parse the csv"))
    for _ in range(4):
        skills.record_outcome("bad", "failure")
    assert skills.search("csv") == []
    assert skills.search("csv", include_quarantined=True)


def test_revive(skills):
    skills.upsert(Skill(name="bad", body="..."))
    for _ in range(4):
        skills.record_outcome("bad", "failure")
    assert skills.revive("bad").status == "active"
