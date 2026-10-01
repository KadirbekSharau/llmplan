"""In-memory traces (M6 uploads): parsed from bytes, never written, content never echoed."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmplan.errors import ValidationError, WorkloadFormatError
from llmplan.workload import InMemoryTrace, load_workload

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("workload_csv_50.csv", "csv"),
        ("workload_azure2023_50.csv", "azure2023"),
        ("workload_azure2024_50.csv", "azure2024"),
        ("workload_burstgpt_50.csv", "burstgpt"),
    ],
)
def test_in_memory_parse_equals_file_parse(name: str, key: str) -> None:
    path = FIXTURES / name
    from_file = load_workload(path)
    from_memory = load_workload(InMemoryTrace("uploaded.csv", path.read_bytes()))
    assert from_memory.format == key
    assert from_memory.source == "uploaded.csv"
    assert from_memory.frame.equals(from_file.frame)
    assert from_memory.notes == from_file.notes


def test_in_memory_errors_name_the_trace_not_its_content() -> None:
    both = b"arrival_s,timestamp,input_tokens,output_tokens\n0,2024-01-01T00:00:00Z,1,1\n"
    with pytest.raises(WorkloadFormatError, match=r"^uploaded\.csv: has both 'arrival_s'"):
        load_workload(InMemoryTrace("uploaded.csv", both))
    with pytest.raises(WorkloadFormatError, match=r"^uploaded\.csv: file is empty"):
        load_workload(InMemoryTrace("uploaded.csv", b""))
    valid = b"arrival_s,input_tokens,output_tokens\n0,1,1\n"
    with pytest.raises(ValidationError, match=r"uploaded\.csv is 43 bytes; the limit is 10 bytes"):
        load_workload(InMemoryTrace("uploaded.csv", valid), max_bytes=10)
    with pytest.raises(WorkloadFormatError, match=r"^uploaded\.csv: cannot parse CSV"):
        load_workload(InMemoryTrace("uploaded.csv", valid + b'"0,1,1\n'), format="csv")
    with pytest.raises(ValidationError, match=r"^cannot read trace uploaded\.csv"):
        load_workload(InMemoryTrace("uploaded.csv", b"\xff\xfe\x00"))


def test_repr_and_str_hide_the_bytes() -> None:
    trace = InMemoryTrace("uploaded.csv", b"secret,rows\n")
    assert "secret" not in repr(trace)
    assert str(trace) == "uploaded.csv"
