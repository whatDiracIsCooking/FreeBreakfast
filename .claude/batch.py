#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic>=1.0"]
# ///
"""Submit a Claude Message Batch, poll it, and collect the results.

The Batches API runs Messages API requests asynchronously at 50% of standard
token prices: up to 100,000 requests per batch, most finishing inside an hour
(24h ceiling), results retrievable for 29 days.

This is a standalone PEP 723 script. `uv run --script` builds it an isolated
environment, so the repo's `pyproject.toml` and `uv.lock` stay untouched -- it
is deliberately not part of the project's Python tooling layer.

    export ANTHROPIC_API_KEY=sk-ant-...     # or `ant auth login`

    uv run --script .claude/batch.py run prompts.jsonl -o results.jsonl

    # or drive the three phases separately
    uv run --script .claude/batch.py submit prompts.jsonl --dry-run
    uv run --script .claude/batch.py submit prompts.jsonl
    uv run --script .claude/batch.py status             # last batch submitted
    uv run --script .claude/batch.py results -o results.jsonl
    uv run --script .claude/batch.py retry              # just the failures

`--max-spend` (default $10) refuses any submission whose *worst case* exceeds
it, so a run that clears the check cannot overspend. The worst case assumes
every request emits its full max_tokens, which is pessimistic by design: the
cheap fix is a realistic `--max-tokens`, not a bigger ceiling.

`--json-schema FILE` constrains every reply to a JSON Schema, so the `text`
field of each result is parseable rather than prose you have to re-read.
`.claude/schemas/audit.json` is one such schema, shaped to the `audit`
skill's findings contract:

    uv run --script .claude/batch.py run docs.jsonl \
        --system CLAUDE.md --json-schema .claude/schemas/audit.json

Input
-----
A `.jsonl` file of one object per line, each either

    {"custom_id": "q1", "prompt": "..."}
    {"custom_id": "q2", "messages": [{"role": "user", "content": "..."}]}

with optional per-request `system`, `model`, `max_tokens` and `effort` keys
overriding the command line. Any other file is read as plain text, one prompt
per line, custom_ids assigned as `req-000000`.

`--system FILE` puts a shared system prompt in front of every request as a
*cached* block, so one large common context is written to cache once and read
at ~0.1x on every request after -- stacking with the 50% batch discount.

Credentials
-----------
The key is never read, logged or stored by this script -- the SDK resolves it
from the environment or an `ant auth login` profile, and `batch-state.json`
holds only batch ids. Every line printed is passed through a redactor as a
backstop, so an `sk-ant-...` string arriving from somewhere we do not control
(an error body, a prompt file) is masked rather than echoed.

Output
------
JSONL, one object per request: custom_id, status, text, stop_reason, usage.
The API returns results in arbitrary order, so they are keyed by custom_id and
never by position. The closing summary reports real spend from the returned
usage, not the pre-flight estimate.
"""

import argparse
import contextlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# (input, output) USD per million tokens, first-party rates. Batches pay half.
# Haiku 5.5's pair is for prompts up to 100K tokens; beyond that it is
# $0.50/$2.50, which this does not model.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.0, 5.0),
}
DEFAULT_MODEL = "claude-opus-5-5"
BATCH_DISCOUNT = 0.5

# A cache read is a multiple of the model's input rate, and the multiple is
# per-model -- it is not a flat 0.1x on the newest models.
CACHE_READ = {
    "claude-fable-5-1": 0.025,
    "claude-mythos-5-1": 0.025,
    "claude-opus-5-5": 0.05,
}
CACHE_READ_DEFAULT = 0.1
# A write costs more for the longer TTL, which is why the TTL is a flag and
# the two are billed apart.
CACHE_WRITE_5M = 1.25
CACHE_WRITE_1H = 2.0
MAX_REQUESTS = 100_000
# The documented cap is 256 MB; stop a little short of it so a batch is never
# rejected over our own framing overhead.
MAX_BYTES = 250_000_000

STATE = Path(__file__).with_name("batch-state.json")


AUTH_HINT = "no usable credentials -- export ANTHROPIC_API_KEY, or run `ant auth login`"

# Nothing here ever reads the key itself -- the SDK resolves it, and the state
# file holds only batch ids. This is the backstop for the paths we do not
# control: an error body, a proxy that echoes a header, a key pasted into a
# prompt file. Everything printed goes through it.
_SECRET = re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{8,}")


def _redact(text: str) -> str:
    return _SECRET.sub("sk-ant-***", text)


def _log(msg: str) -> None:
    print(_redact(msg), file=sys.stderr)


def _client() -> anthropic.Anthropic:
    """Construction never fails for want of a key: the SDK resolves
    credentials lazily at the first request, raising TypeError there. That is
    what `_is_auth_error` and the handlers in `main` are for."""
    return anthropic.Anthropic()


def _is_auth_error(exc: BaseException) -> bool:
    if isinstance(exc, anthropic.AuthenticationError):
        return True
    return isinstance(exc, TypeError) and "authentication method" in str(exc)


# --------------------------------------------------------------------------
# state -- so an interrupted poll never loses the batch id
# --------------------------------------------------------------------------


def _history() -> list[dict]:
    if not STATE.exists():
        return []
    try:
        return json.loads(STATE.read_text())
    except json.JSONDecodeError:
        return []


def _remember(entry: dict) -> None:
    hist = _history()
    hist.append(entry)
    STATE.write_text(json.dumps(hist, indent=2) + "\n")


def _resolve(batch_id: str | None) -> str:
    if batch_id:
        return batch_id
    hist = _history()
    if not hist:
        raise SystemExit(f"no batch id given and none recorded in {STATE}")
    return hist[-1]["id"]


# --------------------------------------------------------------------------
# building requests
# --------------------------------------------------------------------------


def _read_input(path: Path) -> list[dict]:
    if path.suffix in (".jsonl", ".json"):
        objs = []
        for lineno, raw in enumerate(path.read_text().splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: bad JSON: {exc}") from exc
        return objs
    return [{"prompt": ln} for ln in path.read_text().splitlines() if ln.strip()]


def _shared_system(path: Path | None, ttl: str) -> list[dict]:
    """The shared prefix, marked for caching so every request after the first
    reads it instead of paying for it.

    The TTL defaults to 1h here, against the API's own 5m default, because a
    batch is usually processed over more than five minutes -- a 5m entry would
    expire mid-batch and be rewritten repeatedly. The cheap keep-alive for
    that (re-sending with `max_tokens: 0`) is rejected inside a batch, so the
    longer TTL is the only lever. It costs 2x the input rate to write instead
    of 1.25x, and pays for itself from the third read."""
    if path is None:
        return []
    control: dict = {"type": "ephemeral"}
    if ttl != "5m":  # omit for 5m: that is the API's default and always valid
        control["ttl"] = ttl
    return [{"type": "text", "text": path.read_text(), "cache_control": control}]


def _output_format(path: Path | None) -> dict | None:
    """Load `--json-schema`. Accepts either a bare JSON Schema document or a
    ready-made `{"type": "json_schema", "schema": ...}` format object, so a
    schema file is useful on its own."""
    if path is None:
        return None
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: bad JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise SystemExit(f"{path}: expected a JSON object")

    fmt = (
        doc
        if doc.get("type") == "json_schema"
        else {
            "type": "json_schema",
            "schema": doc,
        }
    )
    schema = fmt.get("schema")
    if not isinstance(schema, dict):
        raise SystemExit(f"{path}: no 'schema' object found")
    # Warn rather than refuse: the exact requirement is the API's to enforce,
    # and a stale check here would block a schema that actually works.
    if schema.get("additionalProperties") is not False or "required" not in schema:
        _log(
            f"  note            {path} root has no 'additionalProperties': false"
            " or no 'required' -- the API may reject it"
        )
    return fmt


def _with_ids(objs: list[dict]) -> list[dict]:
    """Stamp on the custom_id each object will be submitted under. Positional
    ids are derived from the input order, so `retry` can match a result back
    to its input line as long as the file has not been reordered."""
    stamped = []
    for i, obj in enumerate(objs):
        copy = dict(obj)
        copy["custom_id"] = str(obj.get("custom_id") or f"req-{i:06d}")
        stamped.append(copy)
    return stamped


def _build(objs: list[dict], args: argparse.Namespace) -> list[Request]:
    shared = _shared_system(args.system, args.cache_ttl)
    fmt = _output_format(args.json_schema)
    requests: list[Request] = []
    seen: set[str] = set()

    for obj in _with_ids(objs):
        cid = obj["custom_id"]
        if cid in seen:
            raise SystemExit(f"duplicate custom_id {cid!r} -- they must be unique")
        seen.add(cid)

        if "messages" in obj:
            messages = obj["messages"]
        elif "prompt" in obj:
            messages = [{"role": "user", "content": obj["prompt"]}]
        else:
            raise SystemExit(f"request {cid}: needs a 'prompt' or 'messages' key")

        params: dict = {
            "model": obj.get("model", args.model),
            "max_tokens": int(obj.get("max_tokens", args.max_tokens)),
            "messages": messages,
        }

        # Shared cached prefix first, per-request text after, so the cacheable
        # prefix stays byte-identical across the batch.
        blocks = [*shared]
        if obj.get("system"):
            blocks.append({"type": "text", "text": obj["system"]})
        if blocks:
            params["system"] = blocks

        # effort and format share output_config, so build it once.
        output_config: dict = {}
        effort = obj.get("effort", args.effort)
        if effort:
            output_config["effort"] = effort
        per_request = obj.get("json_schema")
        if per_request is not None:
            output_config["format"] = {"type": "json_schema", "schema": per_request}
        elif fmt is not None:
            output_config["format"] = fmt
        if output_config:
            params["output_config"] = output_config

        requests.append(
            Request(
                custom_id=cid,
                params=MessageCreateParamsNonStreaming(**params),
            )
        )

    if not requests:
        raise SystemExit("no requests to submit")
    if len(requests) > MAX_REQUESTS:
        raise SystemExit(
            f"{len(requests)} requests exceeds the {MAX_REQUESTS} per-batch cap"
        )

    # The other per-batch cap. Server-side this comes back as an opaque
    # request-too-large error, so measure it here and say which limit it is.
    size = len(json.dumps(requests).encode("utf-8"))
    if size > MAX_BYTES:
        raise SystemExit(
            f"payload is {size / 1e6:.1f} MB, over the "
            f"{MAX_BYTES / 1e6:.0f} MB per-batch cap -- split the input"
        )
    return requests


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------


def _cost(model: str, usage: dict[str, int], fallback: str) -> float | None:
    rates = PRICES.get(model) or PRICES.get(fallback)
    if rates is None:
        return None
    rate_in, rate_out = rates
    read = CACHE_READ.get(model, CACHE_READ_DEFAULT)
    dollars = (
        usage.get("input_tokens", 0) * rate_in
        + usage.get("cache_read_input_tokens", 0) * rate_in * read
        + usage.get("cache_write_5m", 0) * rate_in * CACHE_WRITE_5M
        + usage.get("cache_write_1h", 0) * rate_in * CACHE_WRITE_1H
        + usage.get("output_tokens", 0) * rate_out
    )
    return dollars / 1e6 * BATCH_DISCOUNT


def _count(client: anthropic.Anthropic, params: dict) -> int:
    kwargs = {"model": params["model"], "messages": params["messages"]}
    if "system" in params:
        kwargs["system"] = params["system"]
    return client.messages.count_tokens(**kwargs).input_tokens


def _rough(params: dict) -> int:
    """Offline stand-in for `_count`: ~4 characters per token. Good enough to
    catch an order-of-magnitude mistake, not to price a run."""
    text = json.dumps(params["messages"]) + json.dumps(params.get("system", ""))
    return len(text) // 4


def _estimate(
    client: anthropic.Anthropic, requests: list[Request], sample: int
) -> float | None:
    """Exact token counts on an evenly spread sample, extrapolated. Printed
    before anything is spent; `--dry-run` stops here.

    Returns the worst-case all-in cost -- what `--max-spend` gates on -- or
    None when the model carries no price here."""
    n = len(requests)
    k = max(1, min(sample, n))
    step = max(1, n // k)
    picked = list(range(0, n, step))[:k]
    exact = True
    try:
        counts = [_count(client, requests[i]["params"]) for i in picked]
    except (
        TypeError,
        anthropic.AuthenticationError,
        anthropic.APIConnectionError,
    ) as exc:
        if not _is_auth_error(exc) and not isinstance(
            exc, anthropic.APIConnectionError
        ):
            raise
        exact = False
        counts = [_rough(requests[i]["params"]) for i in picked]
    mean_in = sum(counts) / len(counts)
    out_cap = sum(r["params"]["max_tokens"] for r in requests)
    model = requests[0]["params"]["model"]

    low = _cost(model, {"input_tokens": int(mean_in * n)}, model)
    high = _cost(
        model, {"input_tokens": int(mean_in * n), "output_tokens": out_cap}, model
    )

    _log(f"  requests        {n}")
    _log(f"  model           {model}")
    how = "counted" if exact else "guessed offline"
    _log(f"  sampled         {len(counts)} {how} ({min(counts)}-{max(counts)} tokens)")
    _log(f"  est. input      {int(mean_in * n):,} tokens (mean {mean_in:.0f})")
    _log(f"  output ceiling  {out_cap:,} tokens (sum of max_tokens)")
    if low is None or high is None:
        _log(f"  cost            unknown -- no price on file for {model}")
    else:
        _log(f"  cost            ${low:.2f} input, up to ${high:.2f} all-in")
        _log("                  (batch rate; caching lowers it further)")
    if not exact:
        _log(f"  note            {AUTH_HINT};")
        _log("                  the estimate above is a 4-chars-per-token guess")
    return high


def _check_spend(args: argparse.Namespace, requests: list[Request], high: float | None):
    """`--max-spend` gates on the worst case, so a run that clears it cannot
    overspend whatever the model does. Returns the breach message, or None."""
    if not args.max_spend or high is None or high <= args.max_spend:
        return None
    per = max(r["params"]["max_tokens"] for r in requests)
    return (
        f"worst case ${high:.2f} exceeds the ${args.max_spend:.2f} --max-spend "
        f"ceiling.\nThat is a ceiling, not a forecast: it assumes all "
        f"{len(requests)} requests emit their full {per:,} max_tokens.\n"
        f"Lower --max-tokens to what the replies actually need, cut the "
        f"request count, raise --max-spend, or pass --max-spend 0 to disable."
    )


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def _confirm(args: argparse.Namespace, n: int) -> None:
    if args.yes:
        return
    if not sys.stdin.isatty():
        raise SystemExit("refusing to spend money unattended -- pass --yes")
    reply = input(f"submit {n} requests? [y/N] ").strip().lower()
    if reply not in ("y", "yes"):
        raise SystemExit("aborted")


def _submit(
    args: argparse.Namespace, requests: list[Request], source: str
) -> str | None:
    client = _client()
    high = _estimate(client, requests, args.count_sample)
    breach = _check_spend(args, requests, high)

    if args.dry_run:
        if breach:
            _log(f"\nwould be REFUSED: {breach}")
        _log("\n--dry-run: nothing submitted. First request:")
        print(_redact(json.dumps(requests[0], indent=2)))
        return None
    if breach:
        raise SystemExit(breach)

    _confirm(args, len(requests))
    batch = client.messages.batches.create(requests=requests)
    _remember(
        {
            "id": batch.id,
            "created_at": str(batch.created_at),
            "requests": len(requests),
            "model": args.model,
            "input": source,
        }
    )
    _log(f"submitted {batch.id} ({len(requests)} requests, {batch.processing_status})")
    _log(f"recorded in {STATE}")
    return batch.id


def cmd_submit(args: argparse.Namespace) -> str | None:
    requests = _build(_read_input(args.input), args)
    return _submit(args, requests, str(args.input))


def _retrieve(client: anthropic.Anthropic, batch_id: str):
    try:
        return client.messages.batches.retrieve(batch_id)
    except anthropic.NotFoundError as exc:
        raise SystemExit(f"no such batch: {batch_id}") from exc


# A request that failed for one of these reasons fails again on resubmission:
# the request itself is wrong, or the credential is. Everything else -- an
# overload, a 5xx, an expiry -- is worth another attempt.
PERMANENT = ("invalid_request", "permission", "not_found", "authentication")


def _error_kind(err) -> str:
    """The error classification, probed at both nestings: the batch result
    wraps an error *response* (`{type: "error", error: {type: ...}}`), so the
    useful string can be one or two levels down depending on SDK version."""
    inner = getattr(err, "error", None)
    return getattr(inner, "type", None) or getattr(err, "type", "") or ""


def _failures(
    client: anthropic.Anthropic, batch_id: str
) -> tuple[set[str], list[tuple[str, str]]]:
    """Split an ended batch's non-successes into the worth-retrying and the not."""
    retryable: set[str] = set()
    permanent: list[tuple[str, str]] = []
    for result in client.messages.batches.results(batch_id):
        kind = result.result.type
        if kind == "succeeded":
            continue
        if kind == "errored":
            etype = _error_kind(result.result.error)
            if any(marker in etype for marker in PERMANENT):
                permanent.append((result.custom_id, etype))
                continue
        retryable.add(result.custom_id)
    return retryable, permanent


def _source_of(batch_id: str) -> str | None:
    return next(
        (e.get("input") for e in reversed(_history()) if e["id"] == batch_id), None
    )


def cmd_retry(args: argparse.Namespace) -> None:
    client = _client()
    batch_id = _resolve(args.batch_id)
    batch = _retrieve(client, batch_id)
    if batch.processing_status != "ended":
        raise SystemExit(f"{batch_id} is {batch.processing_status}, not ended")

    source = str(args.input) if args.input else _source_of(batch_id)
    if source is None:
        raise SystemExit(f"no input file recorded for {batch_id} -- pass --input")
    path = Path(source)
    if not path.exists():
        raise SystemExit(f"original input {path} is gone -- pass --input")

    retryable, permanent = _failures(client, batch_id)
    for cid, kind in permanent:
        _log(f"  skipping {cid}: {kind} -- it will fail again, fix the request")
    if not retryable:
        _log(f"nothing retryable in {batch_id}")
        return

    objs = [o for o in _with_ids(_read_input(path)) if o["custom_id"] in retryable]
    missing = retryable - {o["custom_id"] for o in objs}
    if missing:
        raise SystemExit(
            f"{len(missing)} failed ids are not in {path} "
            f"(e.g. {sorted(missing)[:3]}) -- was it edited or reordered?"
        )

    _log(f"retrying {len(objs)} failed requests from {batch_id}")
    _submit(args, _build(objs, args), str(path))


def cmd_status(args: argparse.Namespace) -> None:
    batch = _retrieve(_client(), _resolve(args.batch_id))
    c = batch.request_counts
    print(f"{batch.id}  {batch.processing_status}")
    print(
        f"  processing={c.processing} succeeded={c.succeeded} "
        f"errored={c.errored} canceled={c.canceled} expired={c.expired}"
    )
    print(f"  created={batch.created_at} ended={batch.ended_at}")


def _poll(args: argparse.Namespace, client: anthropic.Anthropic, batch_id: str):
    started = time.monotonic()
    while True:
        batch = _retrieve(client, batch_id)
        if batch.processing_status == "ended":
            return batch
        c = batch.request_counts
        waited = int(time.monotonic() - started)
        _log(
            f"  [{waited:>5}s] {batch.processing_status}: "
            f"processing={c.processing} ok={c.succeeded} err={c.errored}"
        )
        if args.timeout and waited > args.timeout:
            raise SystemExit(
                f"still running after {args.timeout}s. The batch is unaffected "
                f"-- collect it later with `results {batch_id}`."
            )
        time.sleep(args.interval)


def _cache_writes(usage, ttl: str) -> dict[str, int]:
    """Split cache-write tokens by TTL, since the two bill differently. The
    API reports the breakdown under `usage.cache_creation`; when it is absent
    (older SDK, or no cached block) attribute the total to the TTL we asked
    for, which is the one the request actually carried."""
    created = getattr(usage, "cache_creation", None)
    if created is not None:
        return {
            "cache_write_5m": getattr(created, "ephemeral_5m_input_tokens", 0) or 0,
            "cache_write_1h": getattr(created, "ephemeral_1h_input_tokens", 0) or 0,
        }
    total = getattr(usage, "cache_creation_input_tokens", 0) or 0
    key = "cache_write_1h" if ttl == "1h" else "cache_write_5m"
    return {"cache_write_5m": 0, "cache_write_1h": 0, key: total}


def cmd_results(args: argparse.Namespace) -> None:
    client = _client()
    batch_id = _resolve(args.batch_id)
    batch = _retrieve(client, batch_id)
    if batch.processing_status != "ended":
        raise SystemExit(
            f"{batch_id} is {batch.processing_status}, not ended -- "
            "run `status` or `run` to wait for it"
        )

    seen: Counter = Counter()
    totals: Counter = Counter()
    spend = 0.0
    fallback = next(
        (e["model"] for e in reversed(_history()) if e["id"] == batch_id), ""
    )

    with contextlib.ExitStack() as stack:
        fh = sys.stdout if args.out is None else stack.enter_context(args.out.open("w"))
        for result in client.messages.batches.results(batch_id):
            kind = result.result.type
            seen[kind] += 1
            record: dict = {"custom_id": result.custom_id, "status": kind}

            if kind == "succeeded":
                msg = result.result.message
                record["text"] = "".join(
                    b.text for b in msg.content if b.type == "text"
                )
                record["stop_reason"] = msg.stop_reason
                record["model"] = msg.model
                usage = {
                    field: getattr(msg.usage, field, 0) or 0
                    for field in (
                        "input_tokens",
                        "output_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    )
                }
                # `results` can be invoked without the submit flags.
                usage.update(_cache_writes(msg.usage, getattr(args, "cache_ttl", "1h")))
                record["usage"] = usage
                totals.update(usage)
                spend += _cost(msg.model, usage, fallback) or 0.0
                if msg.stop_reason == "max_tokens":
                    seen["truncated"] += 1
                elif msg.stop_reason == "refusal":
                    record["stop_details"] = (
                        msg.stop_details.to_dict() if msg.stop_details else None
                    )
                    seen["refused"] += 1
            elif kind == "errored":
                err = result.result.error
                record["error"] = err.to_dict() if hasattr(err, "to_dict") else str(err)

            fh.write(_redact(json.dumps(record)) + "\n")

    _log("")
    for kind in ("succeeded", "errored", "canceled", "expired"):
        if seen[kind]:
            _log(f"  {kind:<10} {seen[kind]}")
    if seen["truncated"]:
        _log(f"  truncated  {seen['truncated']} (hit max_tokens -- raise it and retry)")
    if seen["refused"]:
        _log(f"  refused    {seen['refused']}")
    if totals:
        _log(
            f"  tokens     {totals['input_tokens']:,} in "
            f"({totals['cache_read_input_tokens']:,} cached) / "
            f"{totals['output_tokens']:,} out"
        )
        _log(f"  spend      ${spend:.2f} at batch rates")
    if args.out is not None:
        _log(f"  written    {args.out}")


def cmd_run(args: argparse.Namespace) -> None:
    batch_id = cmd_submit(args)
    if batch_id is None:
        return
    _log("polling...")
    _poll(args, _client(), batch_id)
    args.batch_id = batch_id
    cmd_results(args)


def cmd_list(args: argparse.Namespace) -> None:
    for i, batch in enumerate(_client().messages.batches.list()):
        if i >= args.limit:
            break
        c = batch.request_counts
        print(
            f"{batch.id}  {batch.processing_status:<10} {batch.created_at}  "
            f"ok={c.succeeded} err={c.errored} proc={c.processing}"
        )


def cmd_cancel(args: argparse.Namespace) -> None:
    batch = _client().messages.batches.cancel(_resolve(args.batch_id))
    print(f"{batch.id}  {batch.processing_status}")


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def _add_request_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument(
        "--effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        help="thinking depth and token spend (default: the model's own)",
    )
    p.add_argument(
        "--system", type=Path, help="file of shared system prompt, sent cached"
    )
    p.add_argument(
        "--cache-ttl",
        choices=("5m", "1h"),
        default="1h",
        help="TTL for the --system cache block. 1h by default, against the "
        "API's 5m: a batch usually runs longer than 5m, and the cheap "
        "keep-alive is rejected inside a batch (writes cost 2x not 1.25x, "
        "and pay off from the third read)",
    )
    p.add_argument(
        "--json-schema",
        type=Path,
        metavar="FILE",
        help="constrain every reply to this JSON Schema (structured outputs). "
        "Takes a bare schema or a {type: json_schema, schema: ...} object; "
        "a per-request 'json_schema' key overrides it",
    )
    p.add_argument(
        "--count-sample",
        type=int,
        default=20,
        help="requests to token-count exactly for the estimate (default 20)",
    )
    p.add_argument(
        "--max-spend",
        type=float,
        default=10.0,
        metavar="USD",
        help="refuse to submit if the WORST CASE exceeds this (0 disables). "
        "The worst case assumes every request emits its full max_tokens, so "
        "lower --max-tokens before raising this (default: 10)",
    )
    p.add_argument("--dry-run", action="store_true", help="estimate, submit nothing")
    p.add_argument("--yes", action="store_true", help="skip the confirmation")


def _add_submit_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", type=Path, help="prompts, as .jsonl or one per line")
    _add_request_flags(p)


def _add_poll_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--interval", type=int, default=30, help="seconds between polls")
    p.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="give up waiting after N seconds (0 = wait; the batch runs on)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="batch.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="create a batch and return")
    _add_submit_flags(p_submit)
    p_submit.set_defaults(func=cmd_submit)

    p_run = sub.add_parser("run", help="submit, poll to completion, collect")
    _add_submit_flags(p_run)
    _add_poll_flags(p_run)
    p_run.add_argument("-o", "--out", type=Path, help="results JSONL (default stdout)")
    p_run.set_defaults(func=cmd_run)

    p_retry = sub.add_parser(
        "retry", help="resubmit the retryable failures of an ended batch"
    )
    p_retry.add_argument("batch_id", nargs="?", help="default: last submitted")
    p_retry.add_argument(
        "--input",
        type=Path,
        help="the original prompts (default: the path recorded at submit)",
    )
    _add_request_flags(p_retry)
    p_retry.set_defaults(func=cmd_retry)

    p_status = sub.add_parser("status", help="one-shot status of a batch")
    p_status.add_argument("batch_id", nargs="?", help="default: last submitted")
    p_status.set_defaults(func=cmd_status)

    p_results = sub.add_parser("results", help="collect an ended batch")
    p_results.add_argument("batch_id", nargs="?", help="default: last submitted")
    p_results.add_argument(
        "-o", "--out", type=Path, help="results JSONL (default stdout)"
    )
    p_results.set_defaults(func=cmd_results)

    p_list = sub.add_parser("list", help="recent batches")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.set_defaults(func=cmd_list)

    p_cancel = sub.add_parser("cancel", help="cancel a running batch")
    p_cancel.add_argument("batch_id", nargs="?", help="default: last submitted")
    p_cancel.set_defaults(func=cmd_cancel)

    args = parser.parse_args()
    try:
        args.func(args)
    except TypeError as exc:
        # The SDK reports unresolvable credentials as a TypeError from deep in
        # the request path. Anything else is a real bug -- let it through.
        if not _is_auth_error(exc):
            raise
        raise SystemExit(AUTH_HINT) from exc
    except anthropic.AuthenticationError as exc:
        # Before APIStatusError, which is its base class.
        raise SystemExit(
            _redact(f"{AUTH_HINT} (the API rejected it: {exc.message})")
        ) from exc
    except anthropic.APIStatusError as exc:
        raise SystemExit(
            _redact(f"API error {exc.status_code}: {exc.message}")
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise SystemExit(f"connection failed: {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit(
            "\ninterrupted -- any submitted batch is still running"
        ) from None


if __name__ == "__main__":
    main()
