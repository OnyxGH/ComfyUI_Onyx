import math
import ast

_SAFE_EVAL_NAMESPACE = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "pow": math.pow,
    "fabs": math.fabs,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "radians": math.radians,
    "degrees": math.degrees,
    "min": min,
    "max": max,
    "abs": abs,
}
_ALLOWED_BINARY_OPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.FloorDiv,
)
_ALLOWED_UNARY_OPS = (ast.UAdd, ast.USub)


def _eval_ast_node(n, variables: dict[str, float]):
    if isinstance(n, ast.Expression):
        return _eval_ast_node(n.body, variables)
    if isinstance(n, ast.Constant):
        if isinstance(n.value, (int, float)):
            return float(n.value)
        raise ValueError("Only numeric constants are allowed")
    if isinstance(n, ast.Num):  # legacy for <3.8
        return float(n.n)  # type: ignore
    if isinstance(n, ast.BinOp) and isinstance(n.op, _ALLOWED_BINARY_OPS):
        left = _eval_ast_node(n.left, variables)
        right = _eval_ast_node(n.right, variables)
        return {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.Mod: lambda a, b: a % b,
            ast.Pow: lambda a, b: a**b,
            ast.FloorDiv: lambda a, b: a // b,
        }[type(n.op)](left, right)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, _ALLOWED_UNARY_OPS):
        operand = _eval_ast_node(n.operand, variables)
        return {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}[type(n.op)](operand)
    if isinstance(n, ast.Name):
        if n.id in variables:
            return float(variables[n.id])
        if n.id in _SAFE_EVAL_NAMESPACE:
            val = _SAFE_EVAL_NAMESPACE[n.id]
            if isinstance(val, (int, float)):
                return float(val)
            return val
        raise ValueError(f"Unknown identifier: {n.id}")
    if isinstance(n, ast.Call):
        if isinstance(n.func, ast.Name) and n.func.id in _SAFE_EVAL_NAMESPACE:
            func = _SAFE_EVAL_NAMESPACE[n.func.id]
        else:
            raise ValueError("Only whitelisted functions are allowed")
        args = [_eval_ast_node(arg, variables) for arg in n.args]
        kwargs = {kw.arg: _eval_ast_node(kw.value, variables) for kw in n.keywords}
        return float(func(*args, **kwargs))  # type: ignore
    raise ValueError("Unsupported expression construct")


def safe_eval(source: str, variables: dict[str, float]) -> float:
    module = ast.parse(source, mode="exec")
    scope = dict(variables)
    last_value: float | None = None

    for statement in module.body:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise ValueError("Only simple variable assignments are allowed")
            value = float(_eval_ast_node(statement.value, scope))
            scope[statement.targets[0].id] = value
            last_value = value
            continue

        if isinstance(statement, ast.Expr):
            last_value = float(_eval_ast_node(statement.value, scope))
            continue

        raise ValueError("Only assignments and expressions are allowed")

    return 0.0 if last_value is None else float(last_value)
