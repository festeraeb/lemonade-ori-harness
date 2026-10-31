use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// One delegated sub-task: a single chat call with a role prompt, no tools.
pub async fn dispatch_subagent(
    client: &reqwest::Client,
    task: &str,
    role: &str,
    model: &str,
    upstream_url: &str,
) -> Result<String, String> {
    if task.trim().is_empty() {
        return Err("task is required".into());
    }
    let sys_prompt = format!(
        "You are an autonomous subagent [role: {}] executing a delegated subtask.\n\
        Analyze the task carefully, execute any needed reasoning, and return the final answer concisely.",
        role
    );
    let messages = [
        json!({ "role": "system", "content": sys_prompt }),
        json!({ "role": "user", "content": task }),
    ];
    let model = if model == "auto" { "default" } else { model };
    let content = crate::llm::chat(client, upstream_url, model, &messages, 800, 0.3).await?;
    Ok(format!("[Subagent '{}' ({})]:\n{}", role, model, content))
}

/// One subagent job in a swarm fan-out (JSON-deserializable from an MCP caller).
#[derive(Debug, Deserialize, Clone)]
pub struct SubagentTask {
    pub task: String,
    #[serde(default = "default_role")]
    pub role: String,
    #[serde(default)]
    pub model: String,
    /// Upstream URL for this subagent; empty => round-robin across the NIM pool.
    #[serde(default)]
    pub upstream: String,
}

fn default_role() -> String {
    "swarm-agent".to_string()
}

/// Pin subagent index `i` to a NIM endpoint round-robin; fall back to `default_url`.
pub fn assign_subagent_url(i: usize, nims: &[String], default_url: &str) -> String {
    if nims.is_empty() {
        default_url.to_string()
    } else {
        nims[i % nims.len()].clone()
    }
}

/// Fan out `tasks` concurrently, bounded by `semaphore` (<= the harness's max-agents,
/// hard cap 5). A task with no model uses `default_model`. Each subagent is pinned to a NIM endpoint round-robin, so the swarm
/// spans the up-to-5 NIM pool ("5 NIM agents through the API"). Returns one result per
/// task, in order.
pub async fn dispatch_subagents(
    client: &reqwest::Client,
    tasks: &[SubagentTask],
    nims: &[String],
    default_url: &str,
    default_model: &str,
    semaphore: Arc<Semaphore>,
) -> Vec<Result<String, String>> {
    use tokio::task::JoinSet;
    let n = tasks.len();
    let mut out: Vec<Result<String, String>> =
        (0..n).map(|_| Err("subagent did not finish".to_string())).collect();
    if n == 0 {
        return out;
    }

    let mut set: JoinSet<(usize, Result<String, String>)> = JoinSet::new();
    for (i, t) in tasks.iter().enumerate() {
        let client = client.clone();
        let t = t.clone();
        let nims = nims.to_vec();
        let default_url = default_url.to_string();
        let default_model = default_model.to_string();
        let sem = semaphore.clone();
        set.spawn(async move {
            // Bound concurrency to the swarm's max-agents (held for the whole subagent turn).
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => return (i, Err("swarm concurrency semaphore closed".to_string())),
            };
            let url = if t.upstream.is_empty() {
                assign_subagent_url(i, &nims, &default_url)
            } else {
                t.upstream.clone()
            };
            let role = if t.role.is_empty() { default_role() } else { t.role.clone() };
            let model = if t.model.is_empty() { default_model } else { t.model.clone() };
            if model.is_empty() {
                return (i, Err("no model for this task".to_string()));
            }
            let res = dispatch_subagent(&client, &t.task, &role, &model, &url).await;
            (i, res)
        });
    }

    while let Some(res) = set.join_next().await {
        match res {
            Ok((i, r)) if i < n => out[i] = r,
            Ok(_) => {}
            // A panicked task leaves its slot as "not scheduled"; say why instead.
            Err(e) => tracing::warn!("swarm subagent task failed: {e}"),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_robin_nim_assignment() {
        let nims = vec!["n1".to_string(), "n2".to_string(), "n3".to_string()];
        assert_eq!(assign_subagent_url(0, &nims, "def"), "n1");
        assert_eq!(assign_subagent_url(1, &nims, "def"), "n2");
        assert_eq!(assign_subagent_url(2, &nims, "def"), "n3");
        assert_eq!(assign_subagent_url(3, &nims, "def"), "n1");
        assert_eq!(assign_subagent_url(4, &nims, "def"), "n2");
        // Empty pool -> default URL.
        assert_eq!(assign_subagent_url(0, &Vec::new(), "def"), "def");
    }
}
