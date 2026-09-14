import io
import json

from rag_chunker import cli


def test_main_writes_jsonl_to_stdout(tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text("# A\n\nHi.\n\n# B\n\nBye.\n")

    status = cli.main([str(doc), "--max-tokens", "512", "--overlap", "0"])

    assert status == 0
    lines = capsys.readouterr().out.strip("\n").split("\n")
    records = [json.loads(line) for line in lines]
    assert [record["heading_path"] for record in records] == [["A"], ["B"]]


def test_main_array_flag_emits_a_single_json_array(tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text("Hello there.")

    status = cli.main([str(doc), "--array"])

    assert status == 0
    parsed = json.loads(capsys.readouterr().out)
    assert isinstance(parsed, list)
    assert len(parsed) == 1


def test_main_no_heading_prefix_drops_prefix_from_text(tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text("# A\n\nHi.\n")

    status = cli.main([str(doc), "--no-heading-prefix"])

    assert status == 0
    record = json.loads(capsys.readouterr().out.strip())
    assert record["text"] == "Hi."


def test_main_stats_flag_prints_summary_to_stderr(tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text("Hello there.")

    status = cli.main([str(doc), "--stats"])

    assert status == 0
    err = capsys.readouterr().err
    assert err.startswith("1 chunks | tokens min")
    assert "oversized" in err


def test_main_output_flag_writes_to_a_file(tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text("Hello there.")
    out_path = tmp_path / "chunks.jsonl"

    status = cli.main([str(doc), "-o", str(out_path)])

    assert status == 0
    assert capsys.readouterr().out == ""
    records = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(records) == 1


def test_main_reads_stdin_when_input_is_dash(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("Hello there."))

    status = cli.main(["-"])

    assert status == 0
    record = json.loads(capsys.readouterr().out.strip())
    assert record["text"] == "Hello there."


def test_main_returns_nonzero_for_invalid_max_tokens(tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text("Hello there.")

    status = cli.main([str(doc), "--max-tokens", "0"])

    assert status == 1
    assert "rag-chunker:" in capsys.readouterr().err


def test_main_returns_nonzero_for_missing_file(tmp_path, capsys):
    missing = tmp_path / "missing.md"

    status = cli.main([str(missing)])

    assert status == 1
    assert "rag-chunker:" in capsys.readouterr().err


def test_main_config_file_supplies_defaults(tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text("Hello there.")
    config = tmp_path / "rag-chunker.json"
    config.write_text(json.dumps({"array": True, "stats": True}))

    status = cli.main([str(doc), "--config", str(config)])

    assert status == 0
    out, err = capsys.readouterr()
    assert isinstance(json.loads(out), list)
    assert "oversized" in err


def test_main_command_line_flag_overrides_config_file(tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text("Hello there.")
    config = tmp_path / "rag-chunker.json"
    config.write_text(json.dumps({"max_tokens": 5}))

    status = cli.main([str(doc), "--config", str(config), "--max-tokens", "512"])

    assert status == 0
    record = json.loads(capsys.readouterr().out.strip())
    assert record["oversized"] is False


def test_main_config_file_rejects_unknown_key(tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text("Hello there.")
    config = tmp_path / "rag-chunker.json"
    config.write_text(json.dumps({"not_a_real_option": True}))

    status = cli.main([str(doc), "--config", str(config)])

    assert status == 1
    assert "not_a_real_option" in capsys.readouterr().err


def test_main_config_file_rejects_wrong_type(tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text("Hello there.")
    config = tmp_path / "rag-chunker.json"
    config.write_text(json.dumps({"max_tokens": "512"}))

    status = cli.main([str(doc), "--config", str(config)])

    assert status == 1
    assert "max_tokens" in capsys.readouterr().err


def test_main_config_file_rejects_bool_for_int_option(tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text("Hello there.")
    config = tmp_path / "rag-chunker.json"
    config.write_text(json.dumps({"overlap": True}))

    status = cli.main([str(doc), "--config", str(config)])

    assert status == 1
    assert "overlap" in capsys.readouterr().err


def test_main_returns_nonzero_for_missing_config_file(tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text("Hello there.")
    missing = tmp_path / "missing.json"

    status = cli.main([str(doc), "--config", str(missing)])

    assert status == 1
    assert "rag-chunker:" in capsys.readouterr().err
