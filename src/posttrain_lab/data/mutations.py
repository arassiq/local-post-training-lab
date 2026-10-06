from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Mutation:
    name: str
    original_code: str
    mutated_code: str


class MutationOperator(Protocol):
    name: str

    def apply(self, code: str) -> Mutation | None: ...


class FlipComparison:
    """Flip the first supported comparison while preserving valid Python."""

    name = "flip-comparison"
    replacements: dict[type[ast.cmpop], type[ast.cmpop]] = {
        ast.GtE: ast.Gt,
        ast.Gt: ast.GtE,
        ast.LtE: ast.Lt,
        ast.Lt: ast.LtE,
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
    }

    def apply(self, code: str) -> Mutation | None:
        tree = ast.parse(code)
        changed = False

        class Transformer(ast.NodeTransformer):
            def visit_Compare(self, node: ast.Compare) -> ast.AST:  # noqa: N802
                nonlocal changed
                self.generic_visit(node)
                if changed:
                    return node
                for index, operator in enumerate(node.ops):
                    replacement = FlipComparison.replacements.get(type(operator))
                    if replacement is not None:
                        node.ops[index] = replacement()
                        changed = True
                        break
                return node

        mutated_tree = Transformer().visit(tree)
        if not changed:
            return None
        ast.fix_missing_locations(mutated_tree)
        return Mutation(self.name, code, ast.unparse(mutated_tree) + "\n")


class FlipBoolean:
    """Flip the first `and`/`or` operator."""

    name = "flip-boolean"

    def apply(self, code: str) -> Mutation | None:
        tree = ast.parse(code)
        changed = False

        class Transformer(ast.NodeTransformer):
            def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:  # noqa: N802
                nonlocal changed
                self.generic_visit(node)
                if not changed:
                    node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
                    changed = True
                return node

        mutated_tree = Transformer().visit(tree)
        if not changed:
            return None
        ast.fix_missing_locations(mutated_tree)
        return Mutation(self.name, code, ast.unparse(mutated_tree) + "\n")
