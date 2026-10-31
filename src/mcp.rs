use crate::policy::{self, Tier};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

/// Most subtasks one spawn_subagents call may fan out.
const MAX_SWARM_TASKS: usize = 20;

/// Tools a research-tier caller never gets.
const OPERATOR_ONLY: &[&str] = &["file_write", "scorecard_eval"];

/// Where action-tier commands start.

fn text(s: impl Into<String>) -> Value {
    json!({ "content": [ { "type": "text", "text": s.into() } ] })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    pub params: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

pub fn list_tools(tier: Tier) -> Value {
    let mut all = all_tools();
    if tier == Tier::Action {
        if let Some(t) = all.get_mut("tools").and_then(|t| t.as_array_mut()).and_then(|a| a.iter_mut().find(|t| t["name"] == "shell_exec")) {
            t["description"] = json!("Run a bash command as the service user under no_new_privs: anything but root (sudo, su, docker, lxd and libvirt are refused; so is ssh back into this machine). timeout_secs up to 900.");
        }
    }
    if tier == Tier::Agent {
        if let Some(tools) = all.get_mut("tools").and_then(|t| t.as_array_mut()) {
            tools.retain(|t| !OPERATOR_ONLY.contains(&t.get("name").and_then(|n| n.as_str()).unwrap_or("")));
            for t in tools.iter_mut() {
                if t.get("name").and_then(|n| n.as_str()) == Some("shell_exec") {
                    t["description"] = json!("Run a read-only command from the agent allowlist (no shell: no ; & > < $ or backticks; pipes allowed). Changes are refused; propose them with buzz_create_task.");
                }
            }
        }
    }
    all
}

fn all_tools() -> Value {
    json!({
        "tools": [
            {
                "name": "shell_exec",
                "description": "Execute a shell command locally with timeout and output capture.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "cmd": { "type": "string", "description": "Command to run" },
                        "timeout_secs": { "type": "number", "description": "Timeout in seconds (default: 30)" }
                    },
                    "required": ["cmd"]
                }
            },
            {
                "name": "file_read",
                "description": "Read content from a local file.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute or relative file path" },
                        "max_bytes": { "type": "number", "description": "Maximum bytes to read" }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "file_write",
                "description": "Write or overwrite content to a local file.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path to write" },
                        "content": { "type": "string", "description": "Content to write" }
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "hardware_status",
                "description": "Inspect local hardware profile, GPU acceleration, and system memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "spawn_subagent",
                "description": "Delegate a sub-task to an autonomous subagent with a dedicated role and model.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": { "type": "string", "description": "Task objective and instructions" },
                        "role": { "type": "string", "description": "Specialized role (coder, researcher, reviewer, tester)" },
                        "model": { "type": "string", "description": "Target model alias or auto" }
                    },
                    "required": ["task"]
                }
            },
            {
                "name": "spawn_subagents",
                "description": "Fan out up to 20 subtasks as one-shot chat calls, at most max-agents (hard cap 5) at once across the whole harness. With a NIM pool configured, subagent i goes to NIM endpoint i mod pool size; otherwise to the gateway.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tasks": { "type": "array", "items": { "type": "object", "description": "Each: task (string, required), role, model, upstream URL (optional)" } },
                        "model": { "type": "string", "description": "Model for tasks that name none. Required when the NIM pool is used; without a pool the gateway's \"auto\" is used" }
                    },
                    "required": ["tasks"]
                }
            },
            {
                "name": "scorecard_eval",
                "description": "Run the Lemonade Challenge benchmark suite and emit a performance scorecard.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "iterations": { "type": "number", "description": "Number of test iterations" },
                        "model": { "type": "string", "description": "Model id (default: the single model the upstream lists)" }
                    }
                }
            },
            {
                "name": "buzz_get_messages",
                "description": "Fetch recent collaborative messages, operator directives, and agent logs from the shared Buzz Board.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": { "type": "number", "description": "Maximum messages to return (default: 20)" },
                        "channel": { "type": "string", "description": "Room, e.g. ops, nodes, models, lab (default: all)" }
                    }
                }
            },
            {
                "name": "buzz_post_message",
                "description": "Post an update, execution plan, or finding to the Buzz Board so operators and other agents see it.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "body": { "type": "string", "description": "Message content to broadcast" },
                        "sender": { "type": "string", "description": "Sender identifier (defaults to Ori-Harness)" },
                        "target": { "type": "string", "description": "Optional target agent or node" },
                        "channel": { "type": "string", "description": "Room (default: the parent's room for a reply, else general)" },
                        "kind": { "type": "string", "description": "note, question, finding, proposal, claim, review, result or status" },
                        "reply_to": { "type": "string", "description": "Id of the message this answers" }
                    },
                    "required": ["body"]
                }
            },
            {
                "name": "buzz_get_tasks",
                "description": "List operational tasks from the Buzz Board. Optionally filter by state (queued, claimed, in_progress, ready_for_review, completed, needs_help).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "state": { "type": "string", "description": "Filter by task state" }
                    }
                }
            },
            {
                "name": "buzz_transition_task",
                "description": "Transition a Buzz Board task state (e.g. claim an unclaimed task, start work, or submit for review).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": { "type": "string", "description": "Task ID e.g. task-..." },
                        "to": { "type": "string", "description": "Target state: claimed, in_progress, ready_for_review, completed, needs_help" },
                        "owner_agent": { "type": "string", "description": "Agent claiming or progressing task (default Ori-Agent)" },
                        "note": { "type": "string", "description": "Context note or execution summary" }
                    },
                    "required": ["task_id", "to"]
                }
            },
            {
                "name": "buzz_create_task",
                "description": "Create a new task on the shared Buzz Board.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": { "type": "string", "description": "Task title" },
                        "description": { "type": "string", "description": "Detailed task description" },
                        "priority": { "type": "number", "description": "Priority (1-100, default 75)" },
                        "owner_node": { "type": "string", "description": "Name of the node that should take the task (optional)" },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "Tags list" }
                    },
                    "required": ["title"]
                }
            },
            {
                "name": "buzz_get_seats_and_roles",
                "description": "Dynamically query live routing seats, role preferences (planner, coder, reviewer, thinker), and node health rather than assuming hardcoded addresses.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "buzz_inbox",
                "description": "Messages addressed to an agent (target or @mention), new posts in its rooms, and open questions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent": { "type": "string", "description": "Agent name" },
                        "rooms": { "type": "string", "description": "Comma-separated rooms (default: rooms the charters make it resident in)" },
                        "since": { "type": "number", "description": "Only messages newer than this, ms since the epoch" }
                    },
                    "required": ["agent"]
                }
            },
            {
                "name": "buzz_react",
                "description": "React to a board message: +1 (useful), verified (you re-ran it and it holds), wrong (it does not hold). Verified results earn their author credit.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message_id": { "type": "string" },
                        "kind": { "type": "string", "description": "+1, verified or wrong" },
                        "by": { "type": "string", "description": "Your agent name" }
                    },
                    "required": ["message_id", "kind", "by"]
                }
            }
        ]
    })
}

/// Environment handed to the MCP surface so tools can reach the NIM pool and the swarm's
/// concurrency semaphore without hard-coded addresses.
#[derive(Clone)]
pub struct HarnessEnv {
    pub gateway: String,
    pub nim_urls: Vec<String>,
    pub buzz: String,
    pub max_agents: usize,
    pub agents_sem: Arc<tokio::sync::Semaphore>,
}

pub async fn call_tool(
    name: &str,
    args: &Value,
    env: &HarnessEnv,
    client: &reqwest::Client,
    tier: Tier,
) -> Result<Value, String> {
    // Gateway as the upstream chat API and the Buzz board URL, as before; the swarm tool
    // reads the rest from `env`.
    let upstream_url = &env.gateway;
    let buzz_url = &env.buzz;
    if tier == Tier::Agent && OPERATOR_ONLY.contains(&name) {
        return Err(format!("{name} is operator-only; propose the change with buzz_create_task"));
    }
    match name {
        "shell_exec" => {
            let cmd = args.get("cmd").and_then(|v| v.as_str()).unwrap_or("");
            if cmd.trim().is_empty() {
                return Err("cmd parameter is required".into());
            }
            let timeout = Duration::from_secs(args.get("timeout_secs").and_then(|v| v.as_u64()).unwrap_or(30).clamp(1, 600));

            if tier == Tier::Action {
                policy::check_action_command(cmd)?;
                let cmd = cmd.to_string();
                let timeout = timeout.min(Duration::from_secs(900));
                let (out, code) = tokio::task::spawn_blocking(move || policy::run_action(&cmd, crate::site::work_dir(), timeout))
                    .await
                    .map_err(|e| e.to_string())??;
                let mut v = text(out);
                v["exit_code"] = json!(code);
                return Ok(v);
            }

            if tier == Tier::Agent {
                let stages = policy::check_command(cmd)?;
                let timeout = timeout.min(Duration::from_secs(60));
                let (out, code) = tokio::task::spawn_blocking(move || policy::run_pipeline(&stages, timeout))
                    .await
                    .map_err(|e| e.to_string())??;
                let mut v = text(out);
                v["exit_code"] = json!(code);
                return Ok(v);
            }

            let mut child = tokio::process::Command::new("bash");
            child.arg("-c").arg(cmd).kill_on_drop(true);
            let output = tokio::time::timeout(timeout, child.output())
                .await
                .map_err(|_| format!("timed out after {}s", timeout.as_secs()))?
                .map_err(|e| format!("failed to execute command: {}", e))?;

            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let combined = if stderr.is_empty() {
                stdout.to_string()
            } else {
                format!("{}\n[stderr]: {}", stdout, stderr)
            };

            Ok(json!({
                "content": [
                    {
                        "type": "text",
                        "text": combined
                    }
                ],
                "exit_code": output.status.code().unwrap_or(-1)
            }))
        }
        "file_read" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let max = args.get("max_bytes").and_then(|v| v.as_u64()).unwrap_or(policy::MAX_OUTPUT as u64) as usize;
            let path = match tier {
                Tier::Agent => policy::check_read_path(path)?.to_string_lossy().to_string(),
                Tier::Action => {
                    policy::check_action_path(path)?;
                    path.to_string()
                }
                Tier::Operator => path.to_string(),
            };
            let bytes = std::fs::read(&path).map_err(|e| format!("read error: {}", e))?;
            let cap = if tier == Tier::Agent { max.min(policy::MAX_OUTPUT) } else { max.max(1) };
            let mut content = String::from_utf8_lossy(&bytes[..bytes.len().min(cap)]).to_string();
            if bytes.len() > cap {
                content.push_str(&format!("\n…(truncated at {cap} of {} bytes)", bytes.len()));
            }
            Ok(text(content))
        }
        "file_write" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
            if tier == Tier::Action {
                policy::check_action_path(path)?;
                if !path.starts_with('/') {
                    return Err("use an absolute path".into());
                }
                if let Some(dir) = std::path::Path::new(path).parent() {
                    let _ = std::fs::create_dir_all(dir);
                }
            }
            std::fs::write(path, content)
                .map_err(|e| format!("write error: {}", e))?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": format!("Successfully written {} bytes to {}", content.len(), path) }
                ]
            }))
        }
        "hardware_status" => {
            let hw = crate::hardware::detect_hardware();
            Ok(json!({
                "content": [
                    { "type": "text", "text": serde_json::to_string_pretty(&hw).unwrap_or_default() }
                ]
            }))
        }
        "spawn_subagent" => {
            let task = args.get("task").and_then(|v| v.as_str()).unwrap_or("");
            let role = args.get("role").and_then(|v| v.as_str()).unwrap_or("worker");
            let model = args.get("model").and_then(|v| v.as_str()).unwrap_or("auto");

            let res = crate::subagent::dispatch_subagent(client, task, role, model, upstream_url).await?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": res }
                ]
            }))
        }
        "spawn_subagents" => {
            let tasks: Vec<crate::subagent::SubagentTask> =
                serde_json::from_value(args.get("tasks").cloned().unwrap_or(Value::Null))
                    .map_err(|e| format!("tasks: {e}"))?;
            if tasks.is_empty() || tasks.len() > MAX_SWARM_TASKS {
                return Err(format!("tasks: give 1 to {MAX_SWARM_TASKS}"));
            }
            let fallback_model = args.get("model").and_then(|v| v.as_str()).unwrap_or("");
            let fallback_model = if !fallback_model.is_empty() {
                fallback_model
            } else if env.nim_urls.is_empty() {
                "auto"
            } else if tasks.iter().all(|t| !t.model.is_empty() || !t.upstream.is_empty()) {
                ""
            } else {
                return Err("model: required when tasks go to the NIM pool (NIM has no \"auto\")".into());
            };
            let results = crate::subagent::dispatch_subagents(
                client, &tasks, &env.nim_urls, upstream_url, fallback_model, env.agents_sem.clone(),
            )
            .await;
            let failed = results.iter().filter(|r| r.is_err()).count();
            let body: Vec<String> = results
                .into_iter()
                .enumerate()
                .map(|(i, r)| match r {
                    Ok(t) => format!("#{i} ok\n{t}"),
                    Err(e) => format!("#{i} error: {e}"),
                })
                .collect();
            Ok(text(format!(
                "{} subagents, {} failed (max {} at once)\n\n{}",
                tasks.len(),
                failed,
                env.max_agents.min(5),
                body.join("\n\n")
            )))
        }
        "scorecard_eval" => {
            let iterations = args.get("iterations").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
            let card = crate::bench::run_benchmark(upstream_url, iterations, args.get("model").and_then(|v| v.as_str()), 64).await?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": serde_json::to_string_pretty(&card).unwrap_or_default() }
                ]
            }))
        }
        "buzz_get_messages" => {
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20).min(200);
            let channel = args.get("channel").and_then(|v| v.as_str()).unwrap_or("all");
            let url = format!("{}/board/messages", buzz_url.trim_end_matches('/'));
            match client.get(&url).query(&[("channel", channel), ("limit", &limit.to_string())]).send().await {
                Ok(resp) => {
                    let val: Value = resp.json().await.map_err(|e| e.to_string())?;
                    let mut msgs = val.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default();
                    if msgs.len() > limit as usize {
                        msgs.truncate(limit as usize);
                    }
                    Ok(json!({
                        "content": [
                            { "type": "text", "text": serde_json::to_string_pretty(&msgs).unwrap_or_default() }
                        ]
                    }))
                }
                Err(e) => Err(format!("failed to fetch buzz messages from {}: {}", url, e)),
            }
        }
        "buzz_post_message" => {
            let body = args.get("body").and_then(|v| v.as_str()).unwrap_or("");
            if body.trim().is_empty() {
                return Err("body parameter is required".into());
            }
            let sender = args.get("sender").and_then(|v| v.as_str()).unwrap_or("Ori-Harness");
            let target = args.get("target").and_then(|v| v.as_str());

            let payload = json!({
                "sender": sender,
                "body": body,
                "target": target,
                "channel": args.get("channel").and_then(|v| v.as_str()),
                "kind": args.get("kind").and_then(|v| v.as_str()),
                "reply_to": args.get("reply_to").and_then(|v| v.as_str())
            });
            let url = format!("{}/board/message", buzz_url.trim_end_matches('/'));
            match client.post(&url).json(&payload).send().await {
                Ok(resp) => {
                    let val: Value = resp.json().await.map_err(|e| e.to_string())?;
                    Ok(json!({
                        "content": [
                            { "type": "text", "text": serde_json::to_string_pretty(&val).unwrap_or_default() }
                        ]
                    }))
                }
                Err(e) => Err(format!("failed to post to buzz board {}: {}", url, e)),
            }
        }
        "buzz_get_tasks" => {
            let state_filter = args.get("state").and_then(|v| v.as_str());
            let url = format!("{}/board/tasks", buzz_url.trim_end_matches('/'));
            match client.get(&url).send().await {
                Ok(resp) => {
                    let val: Value = resp.json().await.map_err(|e| e.to_string())?;
                    let tasks = val.get("tasks").and_then(|t| t.as_array()).cloned().unwrap_or_default();
                    let filtered: Vec<Value> = if let Some(st) = state_filter {
                        tasks.into_iter().filter(|t| t.get("state").and_then(|s| s.as_str()) == Some(st)).collect()
                    } else {
                        tasks
                    };
                    Ok(json!({
                        "content": [
                            { "type": "text", "text": serde_json::to_string_pretty(&filtered).unwrap_or_default() }
                        ]
                    }))
                }
                Err(e) => Err(format!("failed to fetch buzz tasks from {}: {}", url, e)),
            }
        }
        "buzz_transition_task" => {
            let task_id = args.get("task_id").and_then(|v| v.as_str()).unwrap_or("");
            let to = args.get("to").and_then(|v| v.as_str()).unwrap_or("");
            if task_id.is_empty() || to.is_empty() {
                return Err("task_id and to parameters are required".into());
            }
            let owner_agent = args.get("owner_agent").and_then(|v| v.as_str()).unwrap_or("Ori-Agent");
            let note = args.get("note").and_then(|v| v.as_str()).unwrap_or("");

            let payload = json!({
                "task_id": task_id,
                "to": to,
                "owner_agent": owner_agent,
                "note": note
            });
            let url = format!("{}/board/task/transition", buzz_url.trim_end_matches('/'));
            match client.post(&url).json(&payload).send().await {
                Ok(resp) => {
                    let val: Value = resp.json().await.map_err(|e| e.to_string())?;
                    Ok(json!({
                        "content": [
                            { "type": "text", "text": serde_json::to_string_pretty(&val).unwrap_or_default() }
                        ]
                    }))
                }
                Err(e) => Err(format!("failed to transition task {}: {}", task_id, e)),
            }
        }
        "buzz_create_task" => {
            let title = args.get("title").and_then(|v| v.as_str()).unwrap_or("");
            if title.is_empty() {
                return Err("title parameter is required".into());
            }
            let description = args.get("description").and_then(|v| v.as_str()).unwrap_or(title);
            let priority = args.get("priority").and_then(|v| v.as_u64()).unwrap_or(75);
            let owner_node = args.get("owner_node").and_then(|v| v.as_str());
            let tags = args.get("tags").cloned().unwrap_or_else(|| json!(["ori-task"]));

            let payload = json!({
                "title": title,
                "description": description,
                "priority": priority,
                "owner_node": owner_node,
                "tags": tags,
                "sender": args.get("sender").and_then(|v| v.as_str()).unwrap_or("Ori-Agent"),
                "channel": args.get("channel").and_then(|v| v.as_str())
            });
            let url = format!("{}/board/tasks", buzz_url.trim_end_matches('/'));
            match client.post(&url).json(&payload).send().await {
                Ok(resp) => {
                    let val: Value = resp.json().await.map_err(|e| e.to_string())?;
                    Ok(json!({
                        "content": [
                            { "type": "text", "text": serde_json::to_string_pretty(&val).unwrap_or_default() }
                        ]
                    }))
                }
                Err(e) => Err(format!("failed to create task on {}: {}", url, e)),
            }
        }
        "buzz_get_seats_and_roles" => {
            let routing_url = format!("{}/routing", upstream_url.trim_end_matches('/'));
            let status_url = format!("{}/status", buzz_url.trim_end_matches('/'));

            let mut out = json!({
                "status": "ok",
                "upstream": upstream_url,
                "buzz": buzz_url
            });

            if let Ok(resp) = client.get(&routing_url).send().await {
                if let Ok(routing_val) = resp.json::<Value>().await {
                    if let Some(map) = out.as_object_mut() {
                        map.insert("roles".into(), routing_val.get("paddler").and_then(|p| p.get("roles")).cloned().unwrap_or(json!({})));
                        map.insert("seats".into(), routing_val.get("seats").cloned().unwrap_or(json!([])));
                        map.insert("fleet_health".into(), routing_val.get("fleet_health").cloned().unwrap_or(json!({})));
                    }
                }
            }

            if let Ok(resp) = client.get(&status_url).send().await {
                if let Ok(status_val) = resp.json::<Value>().await {
                    if let Some(map) = out.as_object_mut() {
                        map.insert("grid_status".into(), status_val);
                    }
                }
            }

            Ok(json!({
                "content": [
                    { "type": "text", "text": serde_json::to_string_pretty(&out).unwrap_or_default() }
                ]
            }))
        }
        "buzz_inbox" => {
            let agent = args.get("agent").and_then(|v| v.as_str()).unwrap_or("");
            if agent.is_empty() {
                return Err("agent parameter is required".into());
            }
            let mut q: Vec<(&str, String)> = vec![("agent", agent.to_string())];
            if let Some(r) = args.get("rooms").and_then(|v| v.as_str()) {
                q.push(("rooms", r.to_string()));
            }
            if let Some(s) = args.get("since").and_then(|v| v.as_u64()) {
                q.push(("since", s.to_string()));
            }
            let url = format!("{}/board/inbox", buzz_url.trim_end_matches('/'));
            let val: Value = client.get(&url).query(&q).send().await.map_err(|e| e.to_string())?
                .json().await.map_err(|e| e.to_string())?;
            Ok(text(serde_json::to_string_pretty(&val).unwrap_or_default()))
        }
        "buzz_react" => {
            let url = format!("{}/board/react", buzz_url.trim_end_matches('/'));
            let payload = json!({
                "message_id": args.get("message_id").and_then(|v| v.as_str()).unwrap_or(""),
                "kind": args.get("kind").and_then(|v| v.as_str()).unwrap_or(""),
                "by": args.get("by").and_then(|v| v.as_str()).unwrap_or(""),
            });
            let resp = client.post(&url).json(&payload).send().await.map_err(|e| e.to_string())?;
            let ok = resp.status().is_success();
            let val: Value = resp.json().await.map_err(|e| e.to_string())?;
            if !ok {
                return Err(val.get("error").and_then(|e| e.as_str()).unwrap_or("reaction refused").to_string());
            }
            Ok(text(format!("reacted {} to {}", payload["kind"], payload["message_id"])))
        }
        _ => Err(format!("unknown tool '{}'", name)),
    }
}

pub async fn handle_json_rpc(
    req: JsonRpcRequest,
    env: &HarnessEnv,
    client: &reqwest::Client,
    tier: Tier,
) -> JsonRpcResponse {
    let id = req.id;
    match req.method.as_str() {
        "tools/list" => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: Some(list_tools(tier)),
            error: None,
        },
        "tools/call" => {
            let params = req.params.unwrap_or(Value::Null);
            let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args = params.get("arguments").unwrap_or(&Value::Null);

            match call_tool(name, args, env, client, tier).await {
                Ok(res) => JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id,
                    result: Some(res),
                    error: None,
                },
                Err(err) => JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id,
                    result: None,
                    error: Some(json!({ "code": -32000, "message": err })),
                },
            }
        }
        _ => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(json!({ "code": -32601, "message": "Method not found" })),
        },
    }
}

