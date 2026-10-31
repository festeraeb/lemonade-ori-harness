# Lemonade Ori — Standalone Agent Harness

A high-performance, standalone Rust agent harness built on top of **Lemonade** and local inference runtimes. Engineered for local hardware acceleration (Vulkan, Apple Metal, CUDA, CPU), flag-driven mode shifting, first-class Model Context Protocol (MCP), autonomous subagent delegation, and standardized benchmark scorecards for the **Lemonade Challenge**.

---

## ⚡ Quick Start

### 1. Run the Standalone Daemon & Embedded UI
```bash
cargo run --release -- --port 8310
```
Then open `http://127.0.0.1:8310/` in your browser.

### 2. Run the Lemonade Challenge Benchmark
Immediately evaluate local inference throughput and emit a competition scorecard:
```bash
cargo run --release -- --bench
```

### 3. Connect as an MCP Server (Cursor / Claude Desktop / Antigravity)
Add this to your `claude_desktop_config.json` or MCP settings:
```json
{
  "mcpServers": {
    "lemonade-ori": {
      "command": "/path/to/lemonade-ori",
      "args": ["--mcp-stdio"]
    }
  }
}
```

---

## 🚀 Key Features

1. **Hardware Autodetection**: Automatically detects local acceleration:
   - Linux: AMD / Intel Vulkan (`/dev/dri/renderD*`) and NVIDIA CUDA (`nvidia-smi`)
   - macOS: Apple Silicon Metal
   - CPU: Multi-threaded AVX fallback
2. **Flag-Driven Mode Shifting (`--mode`)**:
   - `chat`: Interactive ReAct agent with local tools
   - `plan`: Structured checklist generator with zero hallucinated tool calls
   - `vibe`: High-velocity intuitive architectural flow
   - `eval`: Autonomous single-pass output verifier
   - `bench`: Deterministic Lemonade Challenge QA evaluation
3. **Full Model Context Protocol (MCP) Surface Layer**:
   - HTTP endpoint: `POST /mcp` (JSON-RPC 2.0)
   - STDIO mode: `--mcp-stdio`
   - Built-in tools: `shell_exec`, `file_read`, `file_write`, `hardware_status`, `spawn_subagent`, `spawn_subagents`, `scorecard_eval`
4. **Subagent Orchestration**:
   - Delegate subtasks asynchronously without polluting the parent session's context window.
5. **Zero External Dependencies**:
   - The embedded UI is baked directly into the compiled binary. No node_modules, no python packages, no CDN requirements. Runs 100% offline on any standalone laptop or workstation.

---

## Benchmark (`--bench`)

`--bench` sends real streaming chat requests to the upstream and prints only what it measured or what the upstream reported (`usage`, llama.cpp `timings`). It exits non-zero with the error if any request fails, if the upstream returns no `usage`, or if several models are listed and `--model` is not given. There is no accuracy score: none is computed, so none is printed. `hardware` is not reported as the upstream's; `harness_host_gpus` is the GPUs of the machine running this binary.

One measured run (2026-10-21) against live Lemonade 11.9.0 on a second machine on the LAN (Qwen3-14B-GGUF, served on CPU there), reached through an SSH tunnel. Throughput is that host's, not a claim about any other hardware:

```
ssh -f -N -L 18305:127.0.0.1:13305 lemonade-host
lemonade-ori --bench --upstream http://127.0.0.1:18305/api --model Qwen3-14B-GGUF --bench-iterations 2 --bench-max-tokens 32
```

```json
{
  "worker_id": "localhost-lemonade-ori",
  "backend": "lemonade-ori-rust",
  "runtime": "upstream:http://127.0.0.1:18305/api",
  "harness_host_gpus": [
    "Tesla P100-PCIE-16GB",
    "Tesla P100-PCIE-16GB"
  ],
  "model": "Qwen3-14B-GGUF",
  "role": "lemonade_challenge_runner",
  "runtime_metrics": {
    "iterations": 2,
    "max_tokens": 32,
    "prompt_tokens": 94,
    "completion_tokens": 64,
    "decode_tps": 2.6,
    "decode_source": "upstream_timings",
    "prompt_tps": 116.8,
    "mean_ttft_ms": 1271.6
  },
  "timestamp": "2026-10-21T22:47:03.749764327+00:00"
}
```

Two requests is a smoke test, not a benchmark; `decode_source: upstream_timings` means decode speed came from the upstream's `predicted_ms`.

## Security

The daemon binds to `127.0.0.1` by default. `POST /mcp`, `POST /agent/run` and `GET /agent/runs` need `Authorization: Bearer <token>`, read from `--token-file` (default `~/.config/ori/tokens.json`, mode 600, re-read on every request):

```json
{ "operator": ["<24+ chars>"], "agent": ["<24+ chars>"] }
```

With no token file those routes refuse every request. The stdio MCP mode (`--mcp-stdio`) is unchanged: whoever starts the process already has a shell.

| Token | `shell_exec` | `file_read` | `file_write`, `scorecard_eval` |
|---|---|---|---|
| operator | any command (`bash -c`, `timeout_secs` enforced) | any file | yes |
| agent | read-only allowlist, no shell, see [src/policy.rs](src/policy.rs) | not secrets (`.ssh`, `.config`, `*token*`, `*.key`, …) | refused |

The agent policy is enforced by Ori itself, not by an OS sandbox: `bwrap` cannot create user namespaces on our development host under Ubuntu 24.04's AppArmor default, and loading a profile for it was not approved. So woken agents only read and measure; changes go to forge-pool as Buzz tasks.

## Agent runs (`POST /agent/run`)

One unattended turn for a Buzz room resident, started by a separate waker script that watches the Buzz board (not part of this repo). The model answers with one JSON object per turn (`{"tool": …, "args": …}` or `{"final": …, "kind": …}`), so any chat model works without native tool calling. Tools: allowlisted shell, `read_file`, board messages and tasks, `create_task`, `react`, up to 2 extra posts, up to 3 small-model helper calls. The final answer is posted to the room as a reply to the message that woke it, signed with the agent, model, step count and run id; the full transcript is written to `--runs-dir` (default `~/.local/share/ori/runs`).

Upstreams: `"lemonade"` (default, `--lemonade-url`, local Lemonade on :13305) or `"gateway"` (`--upstream`, the warp gateway on :8126, which streams SSE and may hand a request to a hosted model when a seat is down), or `"nim"` (the `--nim-url` pool; each step of a run goes to the next endpoint in turn). At most `--max-runs` (default 2) run at once.

## Configuration

Everything that describes your machine is read at run time; nothing about one particular network is built in.

| Setting | Default | What it is |
|---|---|---|
| `--token-file` / `ORI_TOKEN_FILE` | `~/.config/ori/tokens.json` | Bearer tokens for `/mcp` and `/agent/*` |
| `--skills-dir` / `ORI_SKILLS_DIR` | `~/.config/ori/skills` | `<name>.md` notes a room can ask for |
| `--runs-dir` / `ORI_RUNS_DIR` | `~/.local/share/ori/runs` | One JSON transcript per agent run |
| `--nim-url` / `ORI_NIM_URL` | `http://127.0.0.1:9135` | OpenAI-compatible NIM endpoints, comma-separated, at most 5; Ori sends no API key, so point them at local proxies that add yours |
| `--max-agents` / `ORI_MAX_AGENTS` | `3` | Subagents at once across the harness for `spawn_subagents` (1 to 5); also the per-run `ask_helper` limit |
| `ORI_HOST_NAME` | the system host name | How agent prompts name this machine |
| `ORI_SELF_HOSTS` | (none) | Extra names and addresses of this machine, comma-separated. Agents may not ssh to them (or to localhost / the host name), since that would leave the no-root sandbox |
| `ORI_WORK_DIR` | `~/ori-work` | Where action agents make git worktrees and run commands |

The action tier runs commands through `setpriv --no-new-privs`, which is Linux-only; on Windows and macOS the harness, MCP server, bench and research tier work, but action runs do not.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

