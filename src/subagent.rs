use serde_json::json;

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
