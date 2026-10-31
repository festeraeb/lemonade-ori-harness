//! One unattended agent run: a woken room resident reads its trigger, uses read-only tools and
//! the board, and posts one answer back to the room (as a reply when there is a trigger).
//!
//! The model talks in plain JSON, one object per turn, so any chat model works without native
//! tool calling:
//!   {"tool": "shell", "args": {"cmd": "df -h /"}}
//!   {"final": "…", "kind": "result"}            (or "post": false to stay silent)

use crate::{llm, mcp, policy::Tier};
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

pub const KINDS: &[&str] = &["note", "question", "finding", "proposal", "claim", "review", "result", "status"];
const MAX_STEPS: u8 = 10;
const MAX_ACTION_STEPS: u8 = 20;
const MAX_EXTRA_POSTS: usize = 2;
const MAX_HELPERS: usize = 5; // hard ceiling; runtime cap is cfg.max_agents.min(MAX_HELPERS)
const OBSERVATION_CHARS: usize = 3000;
const SKILL_CHARS: usize = 1500;

fn yes() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize)]
pub struct RunRequest {
    /// Name the run posts as, e.g. "ori-ops".
    pub agent: String,
    pub room: String,
    /// What to do, in one or two sentences.
    pub goal: String,
    /// The message that woke the agent; the final post replies to it.
    #[serde(default)]
    pub trigger_id: Option<String>,
    /// The thread or digest the waker read, given to the model as context.
    #[serde(default)]
    pub context: Option<String>,
    /// The room charter's purpose line.
    #[serde(default)]
    pub purpose: Option<String>,
    /// Skill folder names under the skills dir; their SKILL.md text is included.
    #[serde(default)]
    pub skills: Vec<String>,
    #[serde(default)]
    pub model: Option<String>,
    /// "lemonade" (local, default) or "gateway" (warp gateway: fleet seats and hosted fallbacks).
    #[serde(default)]
    pub upstream: Option<String>,
    #[serde(default)]
    pub max_steps: Option<u8>,
    /// False: run and return the answer without posting it.
    #[serde(default = "yes")]
    pub post: bool,
    /// "research" (default: read and measure) or "action" (a shell with everything but root).
    /// Action is granted only to NIM models, models over 2B parameters, or models the operator lists.
    #[serde(default)]
    pub tier: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub lemonade: String,
    pub gateway: String,
    /// Up to 5 NIM endpoints (OpenAI-compatible), used round-robin for upstream "nim".
    pub nim_urls: Vec<String>,
    pub buzz: String,
    pub skills_dir: String,
    pub runs_dir: String,
    pub default_model: String,
    pub action_models: Vec<String>,
    /// Max concurrent swarm subagents (operator-tunable; the harness hard-caps at 5).
    pub max_agents: usize,
    /// Shared by every swarm fan-out on this harness, so `max_agents` holds across runs.
    pub agents_sem: std::sync::Arc<tokio::sync::Semaphore>,
}

impl Config {
    /// What the MCP tools need from this config.
    pub fn harness_env(&self) -> mcp::HarnessEnv {
        mcp::HarnessEnv {
            gateway: self.gateway.clone(),
            nim_urls: self.nim_urls.clone(),
            buzz: self.buzz.clone(),
            max_agents: self.max_agents,
            agents_sem: self.agents_sem.clone(),
        }
    }
}

fn valid_agent_name(n: &str) -> Result<(), String> {
    let ok = !n.is_empty()
        && n.len() <= 40
        && n.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        && !n.to_lowercase().starts_with("human")
        && !["forge-pool", "forge-lanes", "buzz-digest", "buzz-stale-report", "buzz-waker"].contains(&n);
    ok.then_some(()).ok_or_else(|| format!("agent name {n:?} is not allowed (letters, digits, -_. ; not a human or a bot name)"))
}

/// The first balanced `{…}` in `text`, parsed. Braces inside strings are skipped.
pub fn extract_json(text: &str) -> Option<Value> {
    let bytes = text.as_bytes();
    let mut start = text.find('{')?;
    loop {
        let (mut depth, mut in_str, mut esc) = (0i32, false, false);
        for (i, &b) in bytes.iter().enumerate().skip(start) {
            if in_str {
                match (esc, b) {
                    (true, _) => esc = false,
                    (false, b'\\') => esc = true,
                    (false, b'"') => in_str = false,
                    _ => {}
                }
                continue;
            }
            match b {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Ok(v) = serde_json::from_str::<Value>(&text[start..=i]) {
                            if v.is_object() {
                                return Some(v);
                            }
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
        start = start + 1 + text[start + 1..].find('{')?;
    }
}

fn skills_text(dir: &str, names: &[String]) -> String {
    names
        .iter()
        .filter(|n| n.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'))
        .filter_map(|n| {
            let t = std::fs::read_to_string(format!("{dir}/{n}/SKILL.md")).ok()?;
            let t: String = t.chars().take(SKILL_CHARS).collect();
            Some(format!("### skill: {n}\n{t}"))
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn action_prompt(req: &RunRequest, skills: &str, nim: &str) -> String {
    format!(
"You are {agent}, an action agent for the #{room} room on the CESARops Buzz board, running unattended on {host}.
Room purpose: {purpose}

You have a real bash shell as user {user} on {host} and can change things: edit files, restart user services, build, run tests, ssh to other machines on your network. Root is off limits: commands run with no_new_privs so sudo cannot work, and sudo, su, docker, lxd, libvirt and ssh back into {host} are refused. If the job needs root, stop and say exactly which command needs root and why.

Rules:
- If you were given a task id: first {{\"tool\": \"task\", \"args\": {{\"task_id\": \"…\", \"to\": \"claimed\", \"note\": \"<plan>\"}}}}, then \"in_progress\". Finish with \"ready_for_review\" (done and verified) or \"blocked\" (with the exact blocker).
- NVIDIA NIM models (chat, code, vision, parse, embeddings) are used by API at {nim}/v1 (OpenAI-compatible; GET /v1/models lists them). \"Use a NIM\" means call it there; do not try to pull or host NIM containers unless the task says so.
- Look before you change: read the file or the service state first. Back up a file before editing it (cp f f.bak-YYYYMMDD).
- Verify every change with a command and report that command and its key output line. Never claim something you did not check.
- Stay inside the task. Do not stop, restart or edit services, models or repos the task does not name; other agents and the operator use them.
- Code changes in a git repo: never switch branches or commit in the shared checkout (others work in it). Make your own worktree: git -C <repo> worktree add {work}/<short-name> -b ori/<short-name>, change and test there, commit there, and say the branch in your final answer; a human merges. Or file a forge task (create_task with tags [\"forge\"] and a ```json {{repo, base_ref, files, test_cmd}} block).
- Be brief: the board is read on a phone.

Reply with exactly ONE JSON object per turn and nothing else:
  {{\"tool\": \"shell\", \"args\": {{\"cmd\": \"systemctl --user status foo\", \"timeout_secs\": 120}}}}
  {{\"tool\": \"read_file\", \"args\": {{\"path\": \"/abs/path\"}}}}
  {{\"tool\": \"write_file\", \"args\": {{\"path\": \"/abs/path\", \"content\": \"…\"}}}}
  {{\"tool\": \"task\", \"args\": {{\"task_id\": \"task-…\", \"to\": \"claimed|in_progress|ready_for_review|blocked\", \"note\": \"…\"}}}}
  {{\"tool\": \"board_messages\", \"args\": {{\"channel\": \"ops\", \"limit\": 20}}}}
  {{\"tool\": \"board_tasks\", \"args\": {{\"state\": \"queued\"}}}}
  {{\"tool\": \"create_task\", \"args\": {{\"title\": \"…\", \"description\": \"…\", \"tags\": [\"…\"]}}}}
  {{\"tool\": \"post\", \"args\": {{\"body\": \"…\", \"kind\": \"question\"}}}}   (at most {extra} extra posts)
  {{\"tool\": \"ask_helper\", \"args\": {{\"task\": \"…\"}}}}   (a small model, at most {helpers} calls)
  {{\"final\": \"what you did, the check you ran and its result\", \"kind\": \"result|finding|note|question\"}}

{skills}",
        agent = req.agent,
        room = req.room,
        purpose = req.purpose.as_deref().unwrap_or("(no charter)"),
        host = crate::site::host_name(),
        user = crate::site::user_name(),
        nim = nim.trim_end_matches('/').trim_end_matches("/v1"),
        work = crate::site::work_dir(),
        extra = MAX_EXTRA_POSTS,
        helpers = MAX_HELPERS,
        skills = if skills.is_empty() { String::new() } else { format!("Skills for this room (follow them):\n{skills}") },
    )
}

fn system_prompt(req: &RunRequest, skills: &str) -> String {
    format!(
"You are {agent}, the resident agent of the #{room} room on the CESARops Buzz board, running unattended on {host}.
Room purpose: {purpose}

Rules:
- You may only read and measure. Commands run without a shell from a read-only allowlist (ls, cat, grep, df, free, nvidia-smi --query-gpu, systemctl status, journalctl, git log/diff/status, curl to fleet addresses, …). Anything that changes the system is refused; propose it instead with create_task.
- Code changes: create_task with tags [\"forge\"] and a description that explains the change and ends with exactly one block
  ```json
  {{\"repo\": \"<absolute path of the git repo>\", \"base_ref\": \"<branch>\", \"files\": [\"<paths the change may touch>\"], \"test_cmd\": \"<command that passes only when the change is right>\"}}
  ```
  forge-pool then builds it on its coding lanes, re-runs test_cmd itself and posts the result to #review; a human merges. Check repo, branch and paths with your tools first; a task without that block fails.
- Never invent output. Report only what a tool returned. If you could not find out, say so.
- Be brief: the board is read on a phone. Lead with the answer, then the command and the key output line.
- Stay silent ({{\"final\": \"\", \"post\": false}}) when you have nothing useful to add.
- Mention another agent or person with @name only when you need something from them.

Reply with exactly ONE JSON object per turn and nothing else:
  {{\"tool\": \"shell\", \"args\": {{\"cmd\": \"df -h /\"}}}}
  {{\"tool\": \"read_file\", \"args\": {{\"path\": \"/abs/path\"}}}}
  {{\"tool\": \"board_messages\", \"args\": {{\"channel\": \"ops\", \"limit\": 20}}}}
  {{\"tool\": \"board_tasks\", \"args\": {{\"state\": \"queued\"}}}}
  {{\"tool\": \"create_task\", \"args\": {{\"title\": \"…\", \"description\": \"…\", \"tags\": [\"ops\"]}}}}
  {{\"tool\": \"react\", \"args\": {{\"message_id\": \"…\", \"kind\": \"verified\"}}}}   (verified only after you re-ran the check yourself; wrong if it does not hold)
  {{\"tool\": \"post\", \"args\": {{\"body\": \"@ori-nodes is the RPC link up?\", \"kind\": \"question\"}}}}   (at most {extra} extra posts)
  {{\"tool\": \"ask_helper\", \"args\": {{\"task\": \"summarise this log: …\"}}}}   (a small model, at most {helpers} calls)
  {{\"final\": \"your answer\", \"kind\": \"result|finding|note|question|proposal\"}}

{skills}",
        agent = req.agent,
        room = req.room,
        purpose = req.purpose.as_deref().unwrap_or("(no charter)"),
        host = crate::site::host_name(),
        extra = MAX_EXTRA_POSTS,
        helpers = MAX_HELPERS,
        skills = if skills.is_empty() { String::new() } else { format!("Skills for this room (follow them):\n{skills}") },
    )
}

fn obs_text(v: &Result<Value, String>) -> String {
    let s = match v {
        Ok(v) => v
            .get("content")
            .and_then(|c| c.as_array())
            .map(|a| a.iter().filter_map(|x| x.get("text").and_then(|t| t.as_str())).collect::<Vec<_>>().join("\n"))
            .unwrap_or_else(|| v.to_string()),
        Err(e) => format!("ERROR: {e}"),
    };
    if s.chars().count() > OBSERVATION_CHARS {
        let cut: String = s.chars().take(OBSERVATION_CHARS).collect();
        format!("{cut}\n…(truncated)")
    } else {
        s
    }
}

fn now_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis()
}

/// The chat endpoint for one step of a run. "nim" rotates across the NIM pool (up to 5
/// endpoints) per step, so one run spreads its calls over the pool.
pub fn resolve_upstream(key: &str, cfg: &Config, step: usize) -> Result<String, String> {
    match key {
        "lemonade" => Ok(cfg.lemonade.clone()),
        "gateway" => Ok(cfg.gateway.clone()),
        "nim" if cfg.nim_urls.is_empty() => Err("upstream \"nim\": no NIM endpoint configured (--nim-url)".into()),
        "nim" => Ok(cfg.nim_urls[step % cfg.nim_urls.len()].clone()),
        other => Err(format!("upstream {other:?}: use \"lemonade\", \"gateway\" or \"nim\"")),
    }
}

pub async fn run(cfg: &Config, client: &reqwest::Client, req: RunRequest) -> Result<Value, String> {
    valid_agent_name(&req.agent)?;
    if req.goal.trim().is_empty() || req.room.trim().is_empty() {
        return Err("goal and room are required".into());
    }
    let upstream_key = req.upstream.as_deref().unwrap_or("lemonade").to_string();
    // Validate the upstream once; "nim" rotates across the pool per step inside the loop.
    resolve_upstream(&upstream_key, cfg, 0)?;
    let model = req.model.clone().unwrap_or_else(|| cfg.default_model.clone());
    // Ori decides the tier from the model, not the caller: a small model asked for action gets research.
    let wants_action = req.tier.as_deref() == Some("action");
    let tier = if wants_action && crate::policy::action_allowed(&model, &cfg.action_models) { Tier::Action } else { Tier::Agent };
    let tier_name = if tier == Tier::Action { "action" } else { "research" };
    let max_steps = if tier == Tier::Action {
        req.max_steps.unwrap_or(12).clamp(1, MAX_ACTION_STEPS)
    } else {
        req.max_steps.unwrap_or(6).clamp(1, MAX_STEPS)
    };
    let run_id = format!("{}-{}", now_ms(), &uuid::Uuid::new_v4().simple().to_string()[..8]);

    let skills = skills_text(&cfg.skills_dir, &req.skills);
    let prompt = if tier == Tier::Action { action_prompt(&req, &skills, cfg.nim_urls.first().map(String::as_str).unwrap_or("")) } else { system_prompt(&req, &skills) };
    let mut messages = vec![
        json!({"role": "system", "content": prompt}),
        json!({"role": "user", "content": format!(
            "Goal: {}\n\n{}",
            req.goal,
            req.context.as_deref().map(|c| format!("Context from the board:\n{c}")).unwrap_or_default()
        )}),
    ];
    let env = cfg.harness_env();
    let mut trace: Vec<Value> = Vec::new();
    if wants_action && tier != Tier::Action {
        trace.push(json!({"note": format!("action tier refused for {model} (needs a NIM model or more than 2B parameters); running as research")}));
    }
    let (mut extra_posts, mut helpers, mut bad_replies) = (0usize, 0usize, 0u8);
    let mut answer: Option<(String, String, bool)> = None;

    let mut llm_error: Option<String> = None;
    messages.push(json!({"role": "user", "content": format!(
        "You have {max_steps} turns in this run. Use them for the most useful part; if the job is bigger, report what you found and what is left.")}));
    for step in 0..max_steps {
        if max_steps >= 4 && step == max_steps - 2 {
            messages.push(json!({"role": "user", "content":
                "Two turns left. Wrap up now: if you hold a task, move it to ready_for_review or blocked with what is done and what is left, then give your final answer with what you found."}));
        }
        // One retry; a model that keeps failing ends the run with a transcript instead of an error.
        // Reasoning models spend tokens thinking before they answer, so leave them room.
        let max_tokens = if tier == Tier::Action { 2500 } else { 900 };
        // Resolve the backend per step: "nim" rotates across the up-to-5 NIM pool.
        let base = resolve_upstream(&upstream_key, cfg, step as usize)?;
        let reply = match llm::chat(client, &base, &model, &messages, max_tokens, 0.2).await {
            Ok(r) => r,
            Err(first) => {
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                match llm::chat(client, &base, &model, &messages, max_tokens, 0.2).await {
                    Ok(r) => r,
                    Err(e) => {
                        trace.push(json!({"step": step, "llm_error": e, "first_error": first}));
                        llm_error = Some(e);
                        break;
                    }
                }
            }
        };
        messages.push(json!({"role": "assistant", "content": reply.clone()}));
        let Some(action) = extract_json(&reply) else {
            bad_replies += 1;
            trace.push(json!({"step": step, "unparsed": reply}));
            if bad_replies > 3 {
                break; // four unusable replies in a row
            }
            messages.push(json!({"role": "user", "content": "That was not a single JSON object. Do not write your reasoning; reply with exactly one JSON object as specified."}));
            continue;
        };
        bad_replies = 0;
        if let Some(fin) = action.get("final") {
            let text = fin.as_str().unwrap_or("").trim().to_string();
            let kind = action.get("kind").and_then(|k| k.as_str()).filter(|k| KINDS.contains(k)).unwrap_or("note").to_string();
            let post = action.get("post").and_then(|p| p.as_bool()).unwrap_or(true) && !text.is_empty();
            trace.push(json!({"step": step, "final": text, "kind": kind, "post": post}));
            answer = Some((text, kind, post));
            break;
        }
        let tool = action.get("tool").and_then(|t| t.as_str()).unwrap_or("");
        let args = action.get("args").cloned().unwrap_or(json!({}));
        let result: Result<Value, String> = match tool {
            "shell" => mcp::call_tool("shell_exec", &args, &env, client, tier).await,
            "read_file" => mcp::call_tool("file_read", &args, &env, client, tier).await,
            "write_file" if tier == Tier::Action => mcp::call_tool("file_write", &args, &env, client, tier).await,
            "task" => {
                let mut a = args.clone();
                a["owner_agent"] = json!(req.agent);
                mcp::call_tool("buzz_transition_task", &a, &env, client, tier).await
            }
            "board_messages" => mcp::call_tool("buzz_get_messages", &args, &env, client, tier).await,
            "board_tasks" => mcp::call_tool("buzz_get_tasks", &args, &env, client, tier).await,
            "create_task" if !crate::policy::action_allowed(&model, &cfg.action_models) => {
                Err("models of 2B parameters or less cannot file tasks; put the suggestion in your final answer".into())
            }
            "create_task" => {
                let mut a = args.clone();
                a["sender"] = json!(req.agent);
                a["channel"] = json!(req.room);
                if tier == Tier::Agent {
                    // A research agent's task is a proposal: the action agent that picks it up judges it first.
                    let mut tags: Vec<Value> = a.get("tags").and_then(|t| t.as_array()).cloned().unwrap_or_default();
                    tags.push(json!("proposal"));
                    a["tags"] = json!(tags);
                }
                mcp::call_tool("buzz_create_task", &a, &env, client, tier).await
            }
            "react" => {
                let mut a = args.clone();
                a["by"] = json!(req.agent);
                mcp::call_tool("buzz_react", &a, &env, client, tier).await
            }
            "post" if extra_posts >= MAX_EXTRA_POSTS => Err(format!("post limit reached ({MAX_EXTRA_POSTS}); put the rest in your final answer")),
            "post" => {
                extra_posts += 1;
                let mut a = args.clone();
                a["sender"] = json!(req.agent);
                if a.get("channel").is_none() {
                    a["channel"] = json!(req.room);
                }
                mcp::call_tool("buzz_post_message", &a, &env, client, tier).await
            }
            "ask_helper" if helpers >= cfg.max_agents.min(MAX_HELPERS) => {
                Err(format!("helper limit reached ({})", cfg.max_agents.min(MAX_HELPERS)))
            }
            "ask_helper" => {
                helpers += 1;
                let task = args.get("task").and_then(|t| t.as_str()).unwrap_or("");
                crate::subagent::dispatch_subagent(client, task, "helper", &cfg.default_model, &cfg.lemonade)
                    .await
                    .map(|t| json!({"content": [{"type": "text", "text": t}]}))
            }
            other => Err(format!("unknown tool {other:?}")),
        };
        let obs = obs_text(&result);
        trace.push(json!({"step": step, "tool": tool, "args": args, "ok": result.is_ok(), "observation": obs}));
        messages.push(json!({"role": "user", "content": format!("Observation from {tool}:\n{obs}")}));
    }

    let mut posted_id: Option<String> = None;
    if let Some((text, kind, true)) = answer.as_ref().map(|(t, k, p)| (t.clone(), k.clone(), *p)) {
        if req.post {
            let body = format!("{text}\n\n— {} ({model}, {tier_name}, {} steps, run {run_id})", req.agent, trace.len());
            let payload = json!({
                "sender": req.agent, "body": body, "channel": req.room, "kind": kind,
                "reply_to": req.trigger_id,
            });
            let url = format!("{}/board/message", cfg.buzz.trim_end_matches('/'));
            let v: Value = client.post(&url).json(&payload).send().await.map_err(|e| e.to_string())?
                .json().await.map_err(|e| e.to_string())?;
            posted_id = v.pointer("/message/id").and_then(|i| i.as_str()).map(str::to_string);
            if posted_id.is_none() {
                return Err(format!("board refused the post: {v}"));
            }
        }
    }

    let record = json!({
        "run_id": run_id, "agent": req.agent, "room": req.room, "goal": req.goal, "trigger_id": req.trigger_id,
        "model": model, "upstream": req.upstream.as_deref().unwrap_or("lemonade"), "tier": tier_name,
        "status": if answer.is_some() { "answered" } else if llm_error.is_some() { "llm_error" } else { "no_final" },
        "final": answer.as_ref().map(|a| a.0.clone()), "kind": answer.as_ref().map(|a| a.1.clone()),
        "posted_id": posted_id, "steps": trace,
    });
    if std::fs::create_dir_all(&cfg.runs_dir).is_ok() {
        let _ = std::fs::write(format!("{}/run-{run_id}.json", cfg.runs_dir), serde_json::to_string_pretty(&record).unwrap_or_default());
    }
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_extraction() {
        assert_eq!(extract_json("sure: {\"final\": \"a {b} c\"} done").unwrap()["final"], "a {b} c");
        assert_eq!(extract_json("{bad} then {\"tool\": \"shell\"}").unwrap()["tool"], "shell");
        assert!(extract_json("no json").is_none());
        assert_eq!(extract_json("{\"a\": {\"b\": 1}}").unwrap()["a"]["b"], 1);
    }

    #[test]
    fn agent_names() {
        assert!(valid_agent_name("ori-ops").is_ok());
        assert!(valid_agent_name("Human-Operator").is_err());
        assert!(valid_agent_name("forge-pool").is_err());
        assert!(valid_agent_name("a b").is_err());
    }
}

#[cfg(test)]
mod upstream_tests {
    use super::*;

    fn cfg(nims: &[&str]) -> Config {
        Config {
            lemonade: "L".into(),
            gateway: "G".into(),
            nim_urls: nims.iter().map(|s| s.to_string()).collect(),
            buzz: String::new(),
            skills_dir: String::new(),
            runs_dir: String::new(),
            default_model: "m".into(),
            action_models: vec![],
            max_agents: 3,
            agents_sem: std::sync::Arc::new(tokio::sync::Semaphore::new(3)),
        }
    }

    #[test]
    fn nim_rotates_per_step_and_needs_a_pool() {
        let c = cfg(&["n1", "n2"]);
        assert_eq!(resolve_upstream("nim", &c, 0).unwrap(), "n1");
        assert_eq!(resolve_upstream("nim", &c, 1).unwrap(), "n2");
        assert_eq!(resolve_upstream("nim", &c, 2).unwrap(), "n1");
        assert_eq!(resolve_upstream("lemonade", &c, 5).unwrap(), "L");
        assert!(resolve_upstream("nim", &cfg(&[]), 0).is_err());
        assert!(resolve_upstream("openrouter", &c, 0).is_err());
    }
}
