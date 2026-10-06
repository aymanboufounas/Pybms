from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import pandas as pd
import pytest
import pybms as pb


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def dataset_server(tmp_path):
    (tmp_path / "data.csv").write_text("a,b,target\n1,2,0\n3,4,1\n", encoding="utf-8")
    (tmp_path / "data.json").write_text(
        '[{"a": 1, "target": 0}, {"a": 3, "target": 1}]', encoding="utf-8"
    )
    (tmp_path / "page.html").write_text("<html>not data</html>", encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize("name", ["iris", "wine", "breast_cancer", "diabetes", "digits"])
def test_offline_datasets(name):
    df = pb.sample_data(name)
    assert len(df) > 50 and "target" in df
    pd.testing.assert_frame_equal(pb.load_data(name), df)


@pytest.mark.parametrize(
    "loader, suffix",
    [
        (pb.from_url, "data.csv?download=1"),
        (pb.load_data, "data.json"),
        (pb.read_csv, "data.csv"),
        (pb.read_json, "data.json"),
    ],
)
def test_url_loading(loader, suffix, dataset_server):
    df = loader(dataset_server + "/" + suffix)
    assert df["a"].tolist() == [1, 3]
    assert df["target"].tolist() == [0, 1]


def test_url_download_failures(dataset_server):
    for url, kwargs in [
        (dataset_server + "/missing.csv", {}),
        (dataset_server + "/page.html", {}),
        (dataset_server + "/data.csv", {"max_bytes": 4}),
        ("file:///secret.csv", {}),
    ]:
        with pytest.raises(pb.DataError):
            pb.from_url(url, **kwargs)


@pytest.mark.parametrize("extension", ["csv", "tsv", "json", "jsonl", "ndjson", "xlsx", "parquet"])
def test_file_roundtrip(extension, tmp_path):
    if extension == "xlsx":
        pytest.importorskip("openpyxl")
    if extension == "parquet":
        pytest.importorskip("pyarrow")
    original = pd.DataFrame({"a": [1, 2], "city": ["Nador", "Fes"]})
    path = pb.save_data(original, tmp_path / f"sample.{extension}")
    restored = (
        pb.read_excel(path)
        if extension == "xlsx"
        else pb.read_parquet(path)
        if extension == "parquet"
        else pb.load_data(path)
    )
    pd.testing.assert_frame_equal(original, restored)


def test_literal_json_and_inmemory_copy():
    df = pb.read_json('[{"x": 1}, {"x": 2}]')
    copied = pb.load_data(df)
    copied.loc[0, "x"] = 999
    assert df.loc[0, "x"] == 1
    with pytest.raises(pb.DataError):
        pb.read_json("[{bad json}]")


@pytest.mark.parametrize(
    "generator", [pb.random_data, pb.classification_data, pb.regression_data, pb.cluster_data]
)
def test_random_generators_are_reproducible(generator):
    first, second = generator(seed=7), generator(seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 100
    assert not first.equals(generator(seed=8))


def test_single_feature_classification():
    assert pb.classification_data(features=1)["target"].nunique() == 2
    with pytest.raises(pb.DataError):
        pb.classification_data(features=1, classes=5)


def test_split_shuffle_batch():
    df = pb.sample_data()
    split = pb.split_data(df, stratify=True)
    a, b, c, d = split
    assert len(a) == len(c) == 120 and len(b) == len(d) == 30
    assert not set(a.index) & set(b.index)
    pd.testing.assert_frame_equal(pb.shuffle_data(df, seed=1), pb.shuffle_data(df, seed=1))
    batches = list(pb.batch_data(df, size=32))
    assert [len(batch) for batch in batches] == [32, 32, 32, 32, 22]
    pd.testing.assert_frame_equal(pd.concat(batches), df)
    with pytest.raises(pb.DataError):
        pb.split_data(df, target="missing")


def test_table_helpers(tmp_path):
    df = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
    assert pb.describe_data(df)["rows"] == 2
    assert pb.preview_data(df, rows=1)["x"].tolist() == [10]
    assert list(pb.select_columns(df, ["x"])) == ["x"]
    joined = pb.join_data(df, pd.DataFrame({"id": [1, 2], "city": ["A", "B"]}), on="id")
    assert joined.shape == (2, 3)
    assert pb.concat_data([df, df]).shape == (4, 2)
    with pytest.raises(pb.DataError):
        pb.join_data(df, pd.DataFrame({"id": [1, 1], "city": ["A", "B"]}), on="id")
    with pytest.raises(pb.DataError):
        pb.select_columns(df, ["absent"])
    assert isinstance(pb.save_data(df, tmp_path / "table.csv"), Path)
