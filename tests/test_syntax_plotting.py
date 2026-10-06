from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import pybms as pb


def test_catalog_is_exactly_100_unique_documented_workflows():
    catalog = pb.list_functions()
    assert len(catalog) == 100 and catalog["name"].is_unique
    for name in catalog["name"]:
        info = pb.function_help(name)
        assert callable(getattr(pb, name)) and info["description"] and "pb." in info["example"]
    assert len(pb.list_functions("Cleaning")) == 20
    assert "auto_train" in pb.list_functions(search="auto_train")["name"].tolist()
    assert "clean_data" in pb.suggest_function("cleen_data")
    with pytest.raises(pb.ConfigurationError):
        pb.function_help("function_that_does_not_exist")
    with pytest.warns(RuntimeWarning):
        assert pb.cleen_data is pb.clean_data


@pytest.mark.parametrize(
    "source",
    [
        "x = [1, 2",
        "print('x'",
        "if True # preserve comment\n    x = 1",
        "def f()\n    return 1",
        "import pybms as pb\nx = pb.cleen_data([])",
    ],
)
def test_syntax_repair(source):
    result = pb.fix_syntax(source)
    assert result.valid and result.changes and pb.check_syntax(result.source)["valid"]


def test_syntax_preserves_strings_comments_and_never_executes(tmp_path):
    path = tmp_path / "should_not_exist"
    source = f"from pathlib import Path\nPath({str(path)!r}).write_text('not executed')\n# pb.cleen_data\ntext = 'pb.cleen_data # [ ]'\n"
    result = pb.fix_syntax(source)
    assert result.source == source and not result.changes and not path.exists()
    assert not pb.fix_syntax("x = 'unterminated").valid
    assert not pb.fix_syntax("x = [1)").valid


def test_syntax_unicode_offsets():
    source = "import pybms as pb\nlabel = 'مرحبا'; df = pb.cleen_data([])"
    result = pb.fix_syntax(source)
    assert "pb.clean_data" in result.source and "مرحبا" in result.source


def test_safe_parameter_typo_repair_and_duplicate_detection():
    with pytest.warns(RuntimeWarning, match="parameters|parameter"):
        df = pb.random_data(rows=3, columns=2, sead=7)
    pd.testing.assert_frame_equal(df, pb.random_data(rows=3, columns=2, seed=7))
    with pytest.raises(TypeError):
        pb.random_data(rows=3, seed=7, sead=8)


def test_cli_help_and_safe_output(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "pybms", "help", "auto_train"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "auto_train" in result.stdout and "cross-validation" not in result.stderr
    source = tmp_path / "bad.py"
    source.write_text("x = [1, 2", encoding="utf-8")
    output = tmp_path / "fixed.py"
    subprocess.run(
        [sys.executable, "-m", "pybms", "fix", str(source), "--output", str(output)],
        check=True,
        capture_output=True,
    )
    assert pb.check_syntax(output.read_text())["valid"] and source.read_text() == "x = [1, 2"
    refused = subprocess.run(
        [sys.executable, "-m", "pybms", "fix", str(source), "--output", str(source)],
        capture_output=True,
    )
    assert refused.returncode != 0 and source.read_text() == "x = [1, 2"


@pytest.mark.parametrize(
    "plotter",
    [
        lambda df: pb.plot(df),
        lambda df: pb.line_plot(df),
        lambda df: pb.scatter_plot(df),
        lambda df: pb.bar_plot(df, x="target"),
        lambda df: pb.hist_plot(df),
        lambda df: pb.box_plot(df),
        lambda df: pb.heatmap(df, annotate=True),
        lambda df: pb.pie_plot(df, "target"),
        lambda df: pb.pair_plot(df, max_columns=2),
        lambda df: pb.missing_plot(df),
        lambda df: pb.class_plot(df),
        lambda df: pb.loss_plot({"loss": [2, 1], "val_loss": [3, 2]}),
        lambda df: pb.confusion_plot([0, 1, 1], [0, 1, 0]),
        lambda df: pb.residual_plot([1, 2], [1.1, 1.9]),
        lambda df: pb.feature_plot({"a": 0.3, "b": 0.7}),
    ],
)
def test_all_plot_types_render(plotter, tmp_path):
    df = pb.classification_data(rows=30, features=2)
    rendered = plotter(df)
    fig = rendered if isinstance(rendered, plt.Figure) else rendered.figure
    path = Path(tmp_path / "plot.png")
    fig.savefig(path)
    assert path.stat().st_size > 1000
    plt.close(fig)


def test_supplied_axes_are_used():
    fig, ax = plt.subplots()
    returned = pb.line_plot(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), ax=ax)
    assert returned is ax and len(ax.lines) == 1
    plt.close(fig)
