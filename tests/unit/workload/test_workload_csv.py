from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llmplan.errors import UnknownRegistryKey, ValidationError, WorkloadFormatError
from llmplan.workload import load_workload
from llmplan.workload.formats import detect, get, keys
from llmplan.workload.formats.generic_csv import write_csv


def write(tmp_path: Path, text: str, name: str = "trace.csv") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_unsorted_arrivals_are_sorted_stably_and_shifted_to_zero(tmp_path: Path) -> None:
    path = write(tmp_path, "arrival_s,input_tokens,output_tokens\n15,3,0\n5,1,1\n15,4,0\n10,2,2\n")
    frame = load_workload(path).frame
    assert frame["arrival_s"].tolist() == [0.0, 5.0, 10.0, 10.0]
    assert frame["input_tokens"].tolist() == [1, 2, 3, 4]


def test_iso_timestamps_with_mixed_offsets(tmp_path: Path) -> None:
    text = (
        "timestamp,input_tokens,output_tokens,model,tenant\n"
        "2024-01-01T00:00:00Z,10,1,m1,t1\n"
        "2024-01-01T02:00:01.5+02:00,20,2,,t2\n"
        "2024-01-01 00:00:02,30,3,m2,\n"
    )
    workload = load_workload(write(tmp_path, text))
    assert workload.format == "csv"
    assert workload.frame["arrival_s"].tolist() == [0.0, 1.5, 2.0]
    assert workload.frame["model"].tolist() == ["m1", pd.NA, "m2"]
    assert workload.frame["tenant"].tolist() == ["t1", "t2", pd.NA]


def test_invalid_rows_are_dropped_and_counted(tmp_path: Path) -> None:
    rows = ["x,1,1", "1,1.5,1", "2,,1", "3,1,-1", "4,99999999999,1", "5,1,abc"]
    good = [f"{i + 10},5,5" for i in range(30)]
    path = write(tmp_path, "arrival_s,input_tokens,output_tokens\n" + "\n".join(rows + good) + "\n")
    workload = load_workload(path)
    assert workload.dropped_rows == 6
    assert len(workload.frame) == 30
    assert workload.notes == ("dropped 6 of 36 rows (16.7%) as invalid",)


def test_few_dropped_rows_add_no_note(tmp_path: Path) -> None:
    rows = ["0,0,1"] + [f"{i},5,5" for i in range(1, 30)]
    workload = load_workload(
        write(tmp_path, "arrival_s,input_tokens,output_tokens\n" + "\n".join(rows))
    )
    assert workload.dropped_rows == 1
    assert workload.notes == ()


def test_write_csv_round_trip_is_exact(tmp_path: Path) -> None:
    text = "arrival_s,input_tokens,output_tokens,model\n0,10,1,a\n0.1,20,2,\n"
    text += "0.30000000000000004,5,0,b\n"
    original = load_workload(write(tmp_path, text))
    out = tmp_path / "out.csv"
    write_csv(original, out)
    assert out.read_text() == text.replace("\n0,", "\n0.0,")
    again = load_workload(out)
    pd.testing.assert_frame_equal(again.frame, original.frame)


def test_write_csv_omits_empty_label_columns(tmp_path: Path) -> None:
    original = load_workload(
        write(tmp_path, "arrival_s,input_tokens,output_tokens,tenant\n0,1,1,\n")
    )
    out = tmp_path / "out.csv"
    write_csv(original, out)
    assert out.read_text() == "arrival_s,input_tokens,output_tokens\n0.0,1,1\n"


def test_utf8_bom_header(tmp_path: Path) -> None:
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbfarrival_s,input_tokens,output_tokens\n0,1,1\n")
    assert detect(path) == "csv"
    assert len(load_workload(path).frame) == 1


@pytest.mark.parametrize(
    ("text", "error", "message"),
    [
        ("", WorkloadFormatError, "empty"),
        ("arrival_s,input_tokens,output_tokens\n", WorkloadFormatError, "no data rows"),
        ("input_tokens,output_tokens\n1,1\n", WorkloadFormatError, "matches no trace format"),
        ('arrival_s,input_tokens,output_tokens\n0,1,1\n1,"2,3\n', WorkloadFormatError, "parse"),
    ],
)
def test_bad_files(tmp_path: Path, text: str, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        load_workload(write(tmp_path, text))


def test_explicit_csv_format_needs_a_time_column(tmp_path: Path) -> None:
    path = write(tmp_path, "input_tokens,output_tokens\n1,1\n")
    with pytest.raises(WorkloadFormatError, match="'arrival_s' or 'timestamp'"):
        load_workload(path, format="csv")


def test_missing_token_column_is_named(tmp_path: Path) -> None:
    path = write(tmp_path, "arrival_s,input_tokens\n0,1\n")
    with pytest.raises(WorkloadFormatError, match="output_tokens"):
        get("csv").parse(path)


def test_missing_file_and_directory(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not a readable file"):
        load_workload(tmp_path / "absent.csv")
    with pytest.raises(ValidationError, match="not a readable file"):
        get("csv").parse(tmp_path)


def test_size_cap(tmp_path: Path) -> None:
    path = write(tmp_path, "arrival_s,input_tokens,output_tokens\n0,1,1\n")
    with pytest.raises(ValidationError, match="limit is 10 bytes"):
        load_workload(path, max_bytes=10)
    with pytest.raises(ValidationError, match="max_bytes"):
        load_workload(path, max_bytes=0)


def test_unknown_format_key(tmp_path: Path) -> None:
    with pytest.raises(UnknownRegistryKey, match="unknown trace format 'parquet'"):
        load_workload(tmp_path / "x", format="parquet")
    assert "csv" in keys()


def test_undecodable_file(tmp_path: Path) -> None:
    path = tmp_path / "binary.csv"
    path.write_bytes(b"\xff\xfe\x00arrival_s\n")
    with pytest.raises(ValidationError, match="cannot read trace"):
        load_workload(path)
