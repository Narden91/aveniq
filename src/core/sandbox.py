import ast
import asyncio
import logging
import multiprocessing as mp
import queue
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from .agents import query_agent as real_query_agent
from .tracing import Tracer

logger = logging.getLogger(__name__)

class _SandboxPrinter:
    """Drop-in replacement for ``print()`` inside the sandbox.
    Captures output in a bounded buffer instead of writing to the host process's stdout."""
    MAX_CHARS = 10_000

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._chars = 0

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        if self._chars >= self.MAX_CHARS:
            return
        text = " ".join(str(a) for a in args)
        self._buffer.append(text)
        self._chars += len(text)

    @property
    def output(self) -> str:
        return "\n".join(self._buffer)


class SandboxSecurityError(Exception):
    """Raised when generated code violates sandbox security constraints."""


class SandboxTimeoutError(Exception):
    """Raised when sandbox execution exceeds the configured timeout."""


@dataclass(frozen=True)
class SandboxPolicy:
    """Execution policy limits used to constrain generated scripts."""

    max_code_chars: int = 30_000
    max_ast_nodes: int = 8_000
    max_statements: int = 1_500
    max_query_calls: int = 120

    def to_dict(self) -> Dict[str, int]:
        return {
            "max_code_chars": self.max_code_chars,
            "max_ast_nodes": self.max_ast_nodes,
            "max_statements": self.max_statements,
            "max_query_calls": self.max_query_calls,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "SandboxPolicy":
        return cls(
            max_code_chars=int(data.get("max_code_chars", cls.max_code_chars)),
            max_ast_nodes=int(data.get("max_ast_nodes", cls.max_ast_nodes)),
            max_statements=int(data.get("max_statements", cls.max_statements)),
            max_query_calls=int(data.get("max_query_calls", cls.max_query_calls)),
        )


_SAFE_BUILTINS = {
    "bool": bool, "bytes": bytes, "complex": complex, "dict": dict, "float": float,
    "frozenset": frozenset, "int": int, "list": list, "set": set, "str": str, "tuple": tuple,
    "abs": abs, "all": all, "any": any, "enumerate": enumerate, "filter": filter,
    "isinstance": isinstance, "len": len, "map": map, "max": max, "min": min,
    "pow": pow, "range": range, "repr": repr, "reversed": reversed, "round": round, "sorted": sorted,
    "sum": sum, "zip": zip, "True": True, "False": False, "None": None,
}

_BLOCKED_AST_NODES = (
    ast.Import, ast.ImportFrom, ast.LShift, ast.RShift, ast.MatMult,
)

_BLOCKED_ATTRIBUTES: Set[str] = {
    "__import__", "__subclasses__", "__bases__", "__mro__", "__globals__", "__code__", "__builtins__", "__class__",
    "__reduce__", "__reduce_ex__", "__qualname__", "__module__", "__dict__", "__weakref__", "__init_subclass__", "__set_name__",
}

_BLOCKED_NAMES: Set[str] = {
    "__import__", "eval", "exec", "compile", "open", "getattr", "setattr", "delattr", "globals", "locals",
    "vars", "dir", "breakpoint", "exit", "quit", "input", "memoryview", "type", "super", "classmethod", "staticmethod",
    "property", "__build_class__",
}


class SpeculativeExecutionTransformer(ast.NodeTransformer):
    """
    Transforms sequential independent await query_agent() calls into asyncio.gather().
    Zero-Latency Speculative Execution Engine.
    """
    def _optimize_block(self, body):
        new_body = []
        batch = []
        
        def flush_batch():
            if not batch: return
            if len(batch) == 1:
                new_body.append(batch[0])
            else:
                targets = []
                calls = []
                for stmt in batch:
                    if isinstance(stmt, ast.Assign):
                        targets.append(stmt.targets[0])
                    else:
                        targets.append(None)
                    calls.append(stmt.value.value)
                
                gather_call = ast.Call(
                    func=ast.Attribute(value=ast.Name(id='asyncio', ctx=ast.Load()), attr='gather', ctx=ast.Load()),
                    args=calls,
                    keywords=[]
                )
                gather_await = ast.Await(value=gather_call)
                
                if all(t is None for t in targets):
                    new_body.append(ast.Expr(value=gather_await))
                else:
                    if any(t is None for t in targets):
                        # complex mix, fallback
                        for b in batch: new_body.append(b)
                        return
                        
                    tup = ast.Tuple(elts=targets, ctx=ast.Store())
                    new_body.append(ast.Assign(targets=[tup], value=gather_await))
            batch.clear()

        for stmt in body:
            is_candidate = False
            assigned_names = set()
            used_names = set()
            
            # Check Assign
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                if isinstance(stmt.value, ast.Await) and isinstance(stmt.value.value, ast.Call):
                    call = stmt.value.value
                    if isinstance(call.func, ast.Name) and call.func.id == 'query_agent':
                        is_candidate = True
                        assigned_names.add(stmt.targets[0].id)
                        for node in ast.walk(call):
                            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                                used_names.add(node.id)
            # Check Expr
            elif isinstance(stmt, ast.Expr):
                if isinstance(stmt.value, ast.Await) and isinstance(stmt.value.value, ast.Call):
                    call = stmt.value.value
                    if isinstance(call.func, ast.Name) and call.func.id == 'query_agent':
                        is_candidate = True
                        for node in ast.walk(call):
                            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                                used_names.add(node.id)
            
            if is_candidate:
                assigned_in_batch = {t.targets[0].id for t in batch if isinstance(t, ast.Assign)}
                if used_names.intersection(assigned_in_batch):
                    flush_batch()
                batch.append(stmt)
            else:
                flush_batch()
                new_body.append(stmt)
                
        flush_batch()
        return new_body

    def visit_FunctionDef(self, node):
        node.body = self._optimize_block(node.body)
        self.generic_visit(node)
        return node
        
    def visit_AsyncFunctionDef(self, node):
        node.body = self._optimize_block(node.body)
        self.generic_visit(node)
        return node
        
    def visit_For(self, node):
        node.body = self._optimize_block(node.body)
        self.generic_visit(node)
        return node
        
    def visit_If(self, node):
        node.body = self._optimize_block(node.body)
        node.orelse = self._optimize_block(node.orelse)
        self.generic_visit(node)
        return node


class CodeSandbox:
    def __init__(
        self,
        timeout_seconds: int = 60,
        isolate_process: bool = True,
        policy: Optional[SandboxPolicy] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.isolate_process = isolate_process
        self.policy = policy or SandboxPolicy()
        # These will be set per execution
        self.tracer: Optional[Tracer] = None
        self.memory: Optional[Any] = None
        self._call_log: Dict[str, str] = {}
        self._pending_query_tasks: Set[asyncio.Task[Any]] = set()
        # Compatibility surface used by tests and introspection
        self.allowed_globals = {"__builtins__": dict(_SAFE_BUILTINS)}
        
    def _make_tracked_query_agent(self):
        """Creates a sandboxed version of query_agent that is traced and logs results."""
        async def _tracked_call(
            agent_type: str,
            prompt: str,
            context_ids: Optional[List[str]] = None,
            weight: Optional[float] = None,
            skill: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            span = self.tracer.start_span(
                name=f"query_agent_{agent_type}",
                span_type="agent",
                inputs={
                    "agent_type": agent_type,
                    "prompt": prompt,
                    "context_ids": context_ids,
                    "weight": weight,
                    "skill": skill,
                }
            )
            try:
                # We return the AgentResult object to the sandbox, so they can use .text
                # If they pass it to str(), it might not work well, so we might need
                # to instruct the orchestrator to use result.text.
                call_kwargs = dict(kwargs)
                if weight is not None:
                    call_kwargs["weight"] = weight
                if skill is not None:
                    call_kwargs["skill"] = skill
                try:
                    result = await real_query_agent(
                        agent_type,
                        prompt,
                        context_ids,
                        **call_kwargs,
                    )
                except TypeError:
                    result = await real_query_agent(
                        agent_type,
                        prompt,
                        context_ids,
                    )

                self._call_log[agent_type] = result.text if hasattr(result, "text") else str(result)
                atoms = [atom.to_dict() for atom in getattr(result, "atoms", [])]
                
                self.tracer.end_span(
                    span,
                    outputs={
                        "text": getattr(result, "text", str(result)),
                        "atom_count": len(atoms),
                        "response_format": getattr(result, "metadata", {}).get("response_format", "plain_text"),
                        "applied_skill": getattr(result, "metadata", {}).get("applied_skill"),
                        "weight": getattr(result, "metadata", {}).get("weight"),
                        "atoms": atoms,
                    },
                    metrics={"tokens": getattr(result, "token_count", 0)},
                )
                return result
            except Exception as e:
                self.tracer.end_span(span, error=e)
                raise

        class _TrackedQueryHandle:
            def __init__(
                self,
                outer: "CodeSandbox",
                agent_type: str,
                prompt: str,
                context_ids: Optional[List[str]] = None,
                weight: Optional[float] = None,
                skill: Optional[str] = None,
                **kwargs: Any,
            ) -> None:
                self._outer = outer
                self._agent_type = agent_type
                self._prompt = prompt
                self._context_ids = context_ids
                self._weight = weight
                self._skill = skill
                self._kwargs = kwargs
                self._task: asyncio.Task[Any] = asyncio.create_task(
                    _tracked_call(
                        self._agent_type,
                        self._prompt,
                        self._context_ids,
                        weight=self._weight,
                        skill=self._skill,
                        **self._kwargs,
                    )
                )
                self._outer._pending_query_tasks.add(self._task)
                self._task.add_done_callback(self._outer._pending_query_tasks.discard)

            def _ensure_task(self) -> asyncio.Task[Any]:
                return self._task

            def __await__(self):
                return self._ensure_task().__await__()

            def done(self) -> bool:
                return self._task is not None and self._task.done()

            def cancelled(self) -> bool:
                return self._task is not None and self._task.cancelled()

            def cancel(self) -> bool:
                return self._ensure_task().cancel()

            def result(self) -> Any:
                return self._ensure_task().result()

            def exception(self) -> BaseException | None:
                return self._ensure_task().exception()

            def add_done_callback(self, callback) -> None:
                self._ensure_task().add_done_callback(callback)

            def remove_done_callback(self, callback) -> int:
                if self._task is None:
                    return 0
                return self._task.remove_done_callback(callback)

            def __getattr__(self, name: str) -> Any:
                if name in {"text", "atoms", "metadata", "token_count", "duration_ms"}:
                    raise AttributeError(
                        "Await query_agent(...) before accessing result fields like .text or .atoms. "
                        "Use task = query_agent(...); result = await task"
                    )
                raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

        def _tracked(
            agent_type: str,
            prompt: str,
            context_ids: Optional[List[str]] = None,
            weight: Optional[float] = None,
            skill: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            return _TrackedQueryHandle(
                self,
                agent_type,
                prompt,
                context_ids,
                weight=weight,
                skill=skill,
                **kwargs,
            )

        return _tracked

    async def _drain_pending_query_tasks(self) -> None:
        pending = [task for task in self._pending_query_tasks if not task.done()]
        self._pending_query_tasks.clear()
        if not pending:
            return

        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    def _make_tracked_memory(self):
        """Creates sandboxed memory functions, traced."""
        def _ensure_memory() -> Any:
            if self.memory is None:
                from .memory import EphemeralMemory

                self.memory = EphemeralMemory()
            return self.memory

        async def _store(key: str, text: str, metadata: dict = None) -> str:
            span = self.tracer.start_span("memory_store", "memory", {"key": key, "text_len": len(text)})
            try:
                mem = _ensure_memory()
                res = await mem.store(key, text, metadata)
                self.tracer.end_span(span, outputs={"key": res})
                return res
            except Exception as e:
                self.tracer.end_span(span, error=e)
                raise

        async def _search(query: str, top_k: int = 5) -> List[dict]:
            span = self.tracer.start_span("memory_search", "memory", {"query": query, "top_k": top_k})
            try:
                mem = _ensure_memory()
                res = await mem.search(query, top_k)
                self.tracer.end_span(span, outputs={"results_count": len(res)})
                return res
            except Exception as e:
                self.tracer.end_span(span, error=e)
                raise
                
        async def _compress(query: str, top_k: int = 5) -> str:
            span = self.tracer.start_span("memory_compress", "memory", {"query": query, "top_k": top_k})
            try:
                mem = _ensure_memory()
                res = await mem.compress_context(query, top_k)
                self.tracer.end_span(span, outputs={"compressed_len": len(res)})
                return res
            except Exception as e:
                self.tracer.end_span(span, error=e)
                raise

        return _store, _search, _compress

    @staticmethod
    def _collect_code_metrics(tree: ast.AST) -> Dict[str, int]:
        ast_nodes = 0
        statements = 0
        query_calls = 0
        for node in ast.walk(tree):
            ast_nodes += 1
            if isinstance(node, ast.stmt):
                statements += 1
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "query_agent"
            ):
                query_calls += 1
        return {
            "ast_nodes": ast_nodes,
            "statements": statements,
            "query_calls": query_calls,
        }

    @staticmethod
    def _validate_and_collect(
        code: str,
        policy: Optional[SandboxPolicy] = None,
    ) -> tuple[ast.Module, Dict[str, int]]:
        active_policy = policy or SandboxPolicy()
        if len(code) > active_policy.max_code_chars:
            raise SandboxSecurityError(
                f"Code size ({len(code)} chars) exceeds policy limit ({active_policy.max_code_chars})."
            )

        tree = ast.parse(code)
        metrics = CodeSandbox._collect_code_metrics(tree)

        if metrics["ast_nodes"] > active_policy.max_ast_nodes:
            raise SandboxSecurityError(
                f"AST size ({metrics['ast_nodes']}) exceeds policy limit ({active_policy.max_ast_nodes})."
            )
        if metrics["statements"] > active_policy.max_statements:
            raise SandboxSecurityError(
                f"Statement count ({metrics['statements']}) exceeds policy limit ({active_policy.max_statements})."
            )
        if metrics["query_calls"] > active_policy.max_query_calls:
            raise SandboxSecurityError(
                f"query_agent calls ({metrics['query_calls']}) exceed policy limit ({active_policy.max_query_calls})."
            )

        has_orchestrate = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise SandboxSecurityError(
                    f"Import statements are not allowed (line {getattr(node, 'lineno', '?')})."
                )
            if isinstance(node, _BLOCKED_AST_NODES):
                raise SandboxSecurityError(f"AST node {type(node).__name__} not allowed (line {getattr(node, 'lineno', '?')})")
            if isinstance(node, ast.Attribute) and node.attr in _BLOCKED_ATTRIBUTES:
                raise SandboxSecurityError(f"Access to '{node.attr}' not allowed (line {getattr(node, 'lineno', '?')})")
            if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
                raise SandboxSecurityError(f"Reference to '{node.id}' not allowed (line {getattr(node, 'lineno', '?')})")
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "orchestrate":
                has_orchestrate = True
        if not has_orchestrate:
            raise ValueError("The generated script must define an 'async def orchestrate():' function.")

        return tree, metrics

    @staticmethod
    def validate_code(code: str, policy: Optional[SandboxPolicy] = None) -> ast.Module:
        tree, _ = CodeSandbox._validate_and_collect(code, policy)
        return tree

    def _execute_in_subprocess(self, code: str) -> Dict[str, Any]:
        ctx = mp.get_context("spawn")
        out_q: Any = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_sandbox_worker_main,
            args=(code, self.timeout_seconds, self.policy.to_dict(), out_q),
        )
        proc.start()
        proc.join(self.timeout_seconds + 2)

        if proc.is_alive():
            proc.terminate()
            proc.join(2)
            raise SandboxTimeoutError(
                f"Orchestration script exceeded the {self.timeout_seconds}s timeout."
            )

        try:
            payload = out_q.get_nowait()
        except queue.Empty:
            if proc.exitcode and proc.exitcode != 0:
                raise RuntimeError(f"Sandbox worker crashed with exit code {proc.exitcode}.")
            raise RuntimeError("Sandbox worker returned no result.")
        finally:
            out_q.close()

        if payload.get("ok"):
            result = payload["result"]
            security = result.get("security", {})
            security["isolation_mode"] = "process"
            security["process_isolated"] = True
            result["security"] = security
            return result

        error_type = payload.get("error_type", "RuntimeError")
        error = payload.get("error", "Sandbox worker execution failed.")

        if error_type == "SandboxSecurityError":
            raise SandboxSecurityError(error)
        if error_type == "SandboxTimeoutError":
            raise SandboxTimeoutError(error)
        raise RuntimeError(error)

    async def _execute_local(self, code: str) -> Dict[str, Any]:
        tree, metrics = self._validate_and_collect(code, self.policy)
        
        # Zero-Latency Speculative Execution Engine Optimization
        tree = SpeculativeExecutionTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        
        # Compile the optimized AST instead of the raw string
        compiled_code = compile(tree, filename="<sandbox>", mode="exec")
        
        # Initialize execution-specific resources
        self._call_log = {}
        self.tracer = Tracer()
        self.memory = None
        self._pending_query_tasks = set()
        sandbox_printer = _SandboxPrinter()
        
        store_fn, search_fn, compress_fn = self._make_tracked_memory()
        
        allowed_globals = {
            "__builtins__": {
                **_SAFE_BUILTINS,
                "print": sandbox_printer,
            },
            "asyncio": asyncio,
            "query_agent": self._make_tracked_query_agent(),
            "memory_store": store_fn,
            "memory_search": search_fn,
            "compress_context": compress_fn
        }

        local_vars: Dict[str, Any] = {}
        try:
            exec(compiled_code, allowed_globals, local_vars)
            orchestrate_func = local_vars['orchestrate']
            if not asyncio.iscoroutinefunction(orchestrate_func):
                raise ValueError("The 'orchestrate' function must be an async function.")

            result = await asyncio.wait_for(orchestrate_func(), timeout=self.timeout_seconds)
            
            return {
                "result": str(result),
                "selected_experts": list(self._call_log.keys()),
                "expert_responses": dict(self._call_log),
                "trace": self.tracer.get_trace(),
                "sandbox_output": sandbox_printer.output,
                "security": {
                    "isolation_mode": "in-process",
                    "process_isolated": False,
                    "policy": self.policy.to_dict(),
                    "observed": metrics,
                },
            }
        except asyncio.TimeoutError:
            logger.error(f"Sandbox execution timed out after {self.timeout_seconds}s")
            raise SandboxTimeoutError(f"Orchestration script exceeded the {self.timeout_seconds}s timeout.")
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            raise
        finally:
            await self._drain_pending_query_tasks()
            if self.memory:
                self.memory.cleanup()

    async def execute(self, code: str) -> Dict[str, Any]:
        if self.isolate_process:
            return await asyncio.to_thread(self._execute_in_subprocess, code)
        return await self._execute_local(code)


def _sandbox_worker_main(
    code: str,
    timeout_seconds: int,
    policy_data: Dict[str, int],
    out_q: Any,
) -> None:
    """Run untrusted orchestration in an isolated worker process."""
    sandbox = CodeSandbox(
        timeout_seconds=timeout_seconds,
        isolate_process=False,
        policy=SandboxPolicy.from_dict(policy_data),
    )
    try:
        result = asyncio.run(sandbox._execute_local(code))
        out_q.put({"ok": True, "result": result})
    except Exception as e:  # pragma: no cover - subprocess transport path
        out_q.put({
            "ok": False,
            "error_type": e.__class__.__name__,
            "error": str(e),
        })
