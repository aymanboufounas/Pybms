"""Conservative source-code repair and searchable function documentation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import difflib
import inspect
import io
import tokenize

from ._registry import REGISTRY, api
from .exceptions import ConfigurationError


@dataclass
class SyntaxResult:
    """Reviewable source, repair descriptions, and final syntax validation."""

    source: str
    changes: list[str]
    valid: bool
    error: dict | None


@api("Syntax & help", "report = pb.check_syntax('x = [1, 2')")
def check_syntax(source: str):
    """Parse Python without executing it; return valid/error/line/offset details."""
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return {"valid": False, "error": exc.msg, "line": exc.lineno, "offset": exc.offset}
    return {"valid": True, "error": None, "line": None, "offset": None}


def _tokens(source):
    result = []
    generator = tokenize.generate_tokens(io.StringIO(source).readline)
    try:
        for token in generator:
            result.append(token)
    except (tokenize.TokenError, IndentationError):
        pass
    return result


def _typos(source, changes):
    tree = ast.parse(source)
    aliases = {
        name.asname or name.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for name in node.names
        if name.name == "pybms"
    }
    replacements = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
        ):
            matches = suggest_function(node.attr, limit=2, cutoff=0.82)
            if node.attr not in REGISTRY and len(matches) == 1:
                # AST column positions are UTF-8 byte offsets, unlike token positions.
                line = source.splitlines(keepends=True)[node.end_lineno - 1]
                end_col = len(line.encode("utf-8")[: node.end_col_offset].decode("utf-8"))
                replacements[(node.end_lineno, end_col - len(node.attr))] = matches[0]
    tokens = _tokens(source)
    for token in tokens:
        if token.type == tokenize.NAME and token.start in replacements:
            changes.append(
                f"Corrected Pybms function {token.string!r} to {replacements[token.start]!r} on line {token.start[0]}."
            )
    changed = [
        token._replace(string=replacements[token.start])
        if token.type == tokenize.NAME and token.start in replacements
        else token
        for token in tokens
    ]
    return tokenize.untokenize(changed) if replacements else source


@api("Syntax & help", "result = pb.fix_syntax('import pybms as pb\nx = pb.array([1, 2]')")
def fix_syntax(source: str, *, repair_brackets=True, repair_colons=True, repair_typos=True):
    """Return repaired source and changes without executing or overwriting anything.

    Supports EOF closing brackets, parser-reported missing header colons, and
    unambiguous top-level Pybms function typos for an explicit import alias.
    Strings/comments are preserved. Quotes, indentation, variable names, and
    ambiguous errors are left for the user. This is an opt-in source helper;
    importing Pybms cannot intercept Python syntax errors in the importing file.
    """
    changes, fixed = [], source
    if repair_brackets and not check_syntax(fixed)["valid"]:
        stack, mismatch = [], False
        for token in _tokens(fixed):
            if token.type != tokenize.OP:
                continue
            if token.string in {"(", "[", "{"}:
                stack.append(token.string)
            elif token.string in {")", "]", "}"}:
                if not stack or {"(": ")", "[": "]", "{": "}"}[stack[-1]] != token.string:
                    mismatch = True
                    break
                stack.pop()
        if stack and not mismatch:
            closers = "".join({"(": ")", "[": "]", "{": "}"}[value] for value in reversed(stack))
            fixed = fixed.rstrip() + "\n" + closers + "\n"
            changes.append(f"Appended missing closing brackets: {closers}.")
    if repair_colons:
        for _ in range(20):
            report = check_syntax(fixed)
            if report["valid"] or report["error"] != "expected ':'" or not report["line"]:
                break
            lines = fixed.splitlines(keepends=True)
            number = report["line"]
            line = lines[number - 1]
            tokens = [
                t
                for t in _tokens(line.lstrip())
                if t.type
                not in {
                    tokenize.INDENT,
                    tokenize.DEDENT,
                    tokenize.NEWLINE,
                    tokenize.NL,
                    tokenize.ENDMARKER,
                }
            ]
            if not tokens or tokens[0].string not in {
                "if",
                "elif",
                "else",
                "for",
                "while",
                "def",
                "class",
                "with",
                "try",
                "except",
                "finally",
                "match",
                "case",
                "async",
            }:
                break
            comment = next((t for t in _tokens(line) if t.type == tokenize.COMMENT), None)
            location = comment.start[1] if comment else len(line.rstrip("\r\n"))
            head, tail = line[:location].rstrip(), line[location:]
            lines[number - 1] = head + ":" + (" " if comment else "") + tail
            fixed = "".join(lines)
            changes.append(f"Added missing header colon on line {number}.")
    if repair_typos and check_syntax(fixed)["valid"]:
        fixed = _typos(fixed, changes)
    result = check_syntax(fixed)
    return SyntaxResult(fixed, changes, result["valid"], None if result["valid"] else result)


@api("Syntax & help", "names = pb.suggest_function('cleen_data')")
def suggest_function(name: str, *, limit=3, cutoff=0.6):
    """Suggest closest documented function names without invoking them."""
    return difflib.get_close_matches(name, sorted(REGISTRY), n=limit, cutoff=cutoff)


@api("Syntax & help", "help_info = pb.function_help('auto_train')")
def function_help(name: str):
    """Return signature, explanation, runnable usage pattern, and category for a function."""
    if name not in REGISTRY:
        raise ConfigurationError(
            f"Unknown function '{name}'. Suggestions: {suggest_function(name)}"
        )
    info = REGISTRY[name]
    return {
        "name": name,
        "category": info["category"],
        "signature": f"{name}{inspect.signature(info['function'])}",
        "description": inspect.getdoc(info["function"]),
        "example": info["example"],
    }


@api("Syntax & help", "catalog = pb.list_functions(category='Cleaning')")
def list_functions(category=None, *, search=None):
    """Return a DataFrame catalog of the 100 workflows; optional category/search filters."""
    import pandas as pd

    rows = []
    for name, info in sorted(REGISTRY.items()):
        doc = inspect.getdoc(info["function"]) or ""
        if category is not None and category.casefold() != info["category"].casefold():
            continue
        if search is not None and search.casefold() not in (name + " " + doc).casefold():
            continue
        rows.append(
            {
                "name": name,
                "category": info["category"],
                "summary": doc.splitlines()[0] if doc else "",
            }
        )
    return pd.DataFrame(rows, columns=["name", "category", "summary"])
