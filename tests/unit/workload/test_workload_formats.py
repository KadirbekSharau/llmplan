from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llmplan.errors import WorkloadFormatError
from llmplan.workload import load_workload
from llmplan.workload.formats import detect, keys

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
AZURE_HEADER = "TIMESTAMP,ContextTokens,GeneratedTokens\n"
BURST_HEADER = "Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "trace.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_registry_keys() -> None:
    assert keys() == ("azure2023", "azure2024", "burstgpt", "csv")


def test_azure2023_relative_seconds_from_naive_timestamps() -> None:
    frame = load_workload(FIXTURES / "workload_azure2023_50.csv").frame
    assert frame["arrival_s"].iloc[1] == pytest.approx(3.855691, abs=1e-9)
    assert frame["input_tokens"].iloc[0] == 4982
    assert frame["model"].isna().all()


@pytest.mark.parametrize(
    ("stamp", "key"),
    [
        ("2024-05-10 00:00:00.009930+00:00", "azure2024"),
        ("2024-05-10T00:00:00Z", "azure2024"),
        ("2024-05-10 00:00:00-0700", "azure2024"),
        ("2023-11-16 18:17:03.9799600", "azure2023"),
    ],
)
def test_azure_release_detected_from_timestamp_style(tmp_path: Path, stamp: str, key: str) -> None:
    assert detect(write(tmp_path, f"{AZURE_HEADER}{stamp},10,2\n")) == key


def test_azure_header_without_rows_is_ambiguous(tmp_path: Path) -> None:
    with pytest.raises(WorkloadFormatError, match="several formats"):
        detect(write(tmp_path, AZURE_HEADER))


def test_azure_short_first_row_reads_as_2023(tmp_path: Path) -> None:
    text = "ContextTokens,GeneratedTokens,TIMESTAMP\n10,2\n5,1,2023-11-16 18:17:03\n"
    assert detect(write(tmp_path, text)) == "azure2023"


def test_azure_unparsable_timestamps_are_dropped(tmp_path: Path) -> None:
    rows = [f"2024-05-10 00:00:{i:02d}.000000+00:00,10,2" for i in range(20)] + ["yesterday,10,2"]
    workload = load_workload(write(tmp_path, AZURE_HEADER + "\n".join(rows) + "\n"))
    assert workload.dropped_rows == 1
    assert workload.frame["arrival_s"].iloc[-1] == 19.0
    assert workload.notes == ()


def test_burstgpt_zero_output_rows_kept_and_noted() -> None:
    workload = load_workload(FIXTURES / "workload_burstgpt_50.csv")
    assert (workload.frame["output_tokens"] == 0).sum() == 2
    assert workload.notes == (
        "zero_output_rows=2: rows with 'Response tokens' == 0 kept (failed or empty)",
    )
    assert set(workload.frame["model"]) == {"ChatGPT", "GPT-4"}
    assert workload.frame["tenant"].isna().all()


def test_burstgpt_newer_columns_are_ignored(tmp_path: Path) -> None:
    text = f"{BURST_HEADER},Session ID,Elapsed time\n5,GPT-4,10,3,13,API log,,1.5\n"
    workload = load_workload(write(tmp_path, text))
    assert workload.format == "burstgpt"
    assert workload.notes == ()
    assert workload.frame["model"].tolist() == ["GPT-4"]


def test_explicit_format_that_does_not_match(tmp_path: Path) -> None:
    path = FIXTURES / "workload_azure2023_50.csv"
    with pytest.raises(WorkloadFormatError, match="requires column"):
        load_workload(path, format="burstgpt")


def test_mixed_headers_are_ambiguous(tmp_path: Path) -> None:
    text = f"{BURST_HEADER},arrival_s,input_tokens,output_tokens\n1,GPT-4,1,1,2,API log,0,1,1\n"
    with pytest.raises(WorkloadFormatError, match="burstgpt, csv"):
        detect(write(tmp_path, text))


def test_generic_csv_fixture_labels() -> None:
    frame = load_workload(FIXTURES / "workload_csv_50.csv").frame
    assert frame["model"].isna().any()
    assert frame["tenant"].isna().any()
    assert set(frame["model"].dropna()) <= {"llama-3-8b", "qwen2.5-7b"}
    assert frame["model"].iloc[0] == "qwen2.5-7b"
    assert frame["tenant"].iloc[1] is pd.NA
