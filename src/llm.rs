//! One chat call to an OpenAI-compatible upstream. Lemonade answers with JSON; the warp
//! gateway (:8126) always streams SSE and may prefix a "↪ handed off to …" notice line when a
//! seat is down. Both come back here as the plain reply text.

use serde_json::{json, Value};

/// Reply text with any `<think>…</think>` block and gateway hand-off lines removed.
pub fn clean(text: &str) -> String {
    let mut t = text.to_string();
    while let (Some(a), Some(b)) = (t.find("<think>"), t.find("</think>")) {
        if b < a {
            break;
        }
        t.replace_range(a..b + "</think>".len(), "");
    }
    t.lines().filter(|l| !l.trim_start().starts_with('↪')).collect::<Vec<_>>().join("\n").trim().to_string()
}

/// Concatenate `choices[0].delta.content` from an SSE body.
pub fn from_sse(body: &str) -> String {
    let mut out = String::new();
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data:") else { continue };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        if let Ok(v) = serde_json::from_str::<Value>(data) {
            if let Some(s) = v.pointer("/choices/0/delta/content").and_then(|c| c.as_str()) {
                out.push_str(s);
            } else if let Some(s) = v.pointer("/choices/0/message/content").and_then(|c| c.as_str()) {
                out.push_str(s);
            }
        }
    }
    out
}

pub async fn chat(
    client: &reqwest::Client,
    base: &str,
    model: &str,
    messages: &[Value],
    max_tokens: u32,
    temperature: f32,
) -> Result<String, String> {
    let url = format!("{}/v1/chat/completions", base.trim_end_matches('/'));
    let body = json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": false,
    });
    let resp = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(300))
        .send()
        .await
        .map_err(|e| format!("{url}: {e}"))?;
    let status = resp.status();
    let text = resp.text().await.map_err(|e| format!("{url}: {e}"))?;
    if !status.is_success() {
        return Err(format!("{url}: HTTP {status}: {}", text.chars().take(300).collect::<String>()));
    }
    let raw = match serde_json::from_str::<Value>(&text) {
        Ok(v) => {
            let msg = v.pointer("/choices/0/message").ok_or_else(|| format!("{url}: no choices[0].message in reply"))?;
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            // Reasoning models sometimes put the answer in reasoning_content and leave content empty; use it
            // only when it holds a JSON object (an action), not plain thinking.
            if content.trim().is_empty() {
                let r = msg.get("reasoning_content").and_then(|c| c.as_str()).unwrap_or("");
                if r.contains("{\"tool\"") || r.contains("{\"final\"") { r.to_string() } else { String::new() }
            } else {
                content.to_string()
            }
        }
        Err(_) => from_sse(&text),
    };
    let reply = clean(&raw);
    if reply.is_empty() {
        return Err(format!("{url}: empty reply from {model}"));
    }
    Ok(reply)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_and_cleaning() {
        let body = ": warp-gateway connecting\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"\u{21aa} seat down \u{2014} handed off\\n\\n\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"<think>hmm</think>{\\\"final\\\"\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\": \\\"ok\\\"}\"}}]}\n\ndata: [DONE]\n";
        assert_eq!(clean(&from_sse(body)), "{\"final\": \"ok\"}");
    }
}
