"""Arithmetic the model should not be doing in its head.

Implemented as an AST walk over a whitelist of node types, not ``eval``. The
input string is model-generated and therefore untrusted: anything outside plain
arithmetic — attribute access, calls, names, comprehensions — is rejected
before evaluation rather than sandboxed after it.
"""

from __future__ import annotations

import ast
import math
import operator

from langchain_core.tools import tool

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

MAX_EXPONENT = 64
MAX_EXPRESSION_CHARS = 200


class CalculatorError(ValueError):
    pass


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError(f"unsupported literal: {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"unsupported operator: {type(node.op).__name__}")
        left, right = _eval(node.left), _eval(node.right)
        # Bound exponentiation: 9**9**9 is a denial of service, not a sum.
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError(f"exponent above {MAX_EXPONENT} rejected")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise CalculatorError("division by zero")
        return float(op(left, right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"unsupported unary operator: {type(node.op).__name__}")
        return float(op(_eval(node.operand)))
    raise CalculatorError(f"unsupported expression element: {type(node).__name__}")


def evaluate(expression: str) -> float:
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalculatorError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"could not parse: {exc.msg}") from exc
    value = _eval(tree)
    if math.isnan(value) or math.isinf(value):
        raise CalculatorError("result is not a finite number")
    return value


@tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a plain arithmetic expression, e.g. "(1240 - 890) / 890 * 100".

    Supports + - * / // % ** and parentheses. No variables, no functions.
    """
    try:
        return f"{expression} = {evaluate(expression):.6g}"
    except CalculatorError as exc:
        return f"CALCULATOR ERROR: {exc}"
