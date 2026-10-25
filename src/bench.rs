//! Honest benchmark: every number in the scorecard is either reported by the
//! upstream (`usage` / `timings` in the stream) or measured here with a wall
//! clock. If a request fails or the upstream returns no usable usage, the
//! whole benchmark fails with an error; nothing is defaulted or simulated.
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scorecard {
    pub worker_id: String,
    pub backend: String,
    pub runtime: String,
    /// GPUs detected on the machine running this harness, NOT necessarily the upstream's hardware.
    pub harness_host_gpus: Vec<String>,
    pub model: String,
    pub role: String,
    pub runtime_metrics: RuntimeMetrics,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeMetrics {
    /// Completed requests (equals the requested count, or the run errors).
    pub iterations: usize,
    pub max_tokens: u32,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// completion tokens / decode seconds (see `decode_source`).
    pub decode_tps: f64,
    /// "upstream_timings" if every run reported predicted_ms, else "wall_clock".
    pub decode_source: String,
    /// Only present when the upstream reported prompt_per_second/prompt_ms for every run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tps: Option<f64>,
    /// Wall time from sending the request to the first streamed content delta.
    pub mean_ttft_ms: f64,
}

/// What one completed request produced.
#[derive(Debug, Clone, PartialEq)]
pub struct RunResult {
    pub ttft_ms: f64,
    pub wall_ms: f64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub upstream_predicted_ms: Option<f64>,
    pub upstream_prompt_ms: Option<f64>,
}

/// Pure aggregation over completed runs. Errors on anything unusable.
pub fn aggregate(runs: &[RunResult], max_tokens: u32) -> Result<RuntimeMetrics, String> {
    if runs.is_empty() {
        return Err("no completed requests".into());
    }
    let mut prompt_tokens = 0u64;
    let mut completion_tokens = 0u64;
    let mut decode_secs = 0.0;
    let mut ttft_sum = 0.0;
    let mut all_timings = true;
    let mut prompt_secs = 0.0;
    let mut all_prompt_timings = true;
    for (i, r) in runs.iter().enumerate() {
        if r.completion_tokens == 0 {
            return Err(format!("run {}: upstream reported 0 completion tokens", i + 1));
        }
        prompt_tokens += r.prompt_tokens;
        completion_tokens += r.completion_tokens;
        ttft_sum += r.ttft_ms;
        match r.upstream_predicted_ms {
            Some(ms) if ms > 0.0 => decode_secs += ms / 1000.0,
            _ => {
                all_timings = false;
                decode_secs += (r.wall_ms - r.ttft_ms) / 1000.0;
            }
        }
        match r.upstream_prompt_ms {
            Some(ms) if ms > 0.0 && r.prompt_tokens > 0 => prompt_secs += ms / 1000.0,
            _ => all_prompt_timings = false,
        }
    }
    if decode_secs <= 0.0 {
        return Err("decode time is zero or negative; cannot compute decode_tps".into());
    }
    let round1 = |x: f64| (x * 10.0).round() / 10.0;
    Ok(RuntimeMetrics {
        iterations: runs.len(),
        max_tokens,
        prompt_tokens,
        completion_tokens,
        decode_tps: round1(completion_tokens as f64 / decode_secs),
        decode_source: if all_timings { "upstream_timings" } else { "wall_clock" }.into(),
        prompt_tps: if all_prompt_timings && prompt_secs > 0.0 {
            Some(round1(prompt_tokens as f64 / prompt_secs))
        } else {
            None
        },
        mean_ttft_ms: round1(ttft_sum / runs.len() as f64),
    })
}

/// Use `requested` if given; otherwise the upstream's /v1/models list, which
/// must contain exactly one model (otherwise the caller must choose).
pub async fn resolve_model(
    client: &reqwest::Client,
    upstream_url: &str,
    requested: Option<&str>,
) -> Result<String, String> {
    if let Some(m) = requested {
        return Ok(m.to_string());
    }
    let url = format!("{}/v1/models", upstream_url.trim_end_matches('/'));
    let resp = client.get(&url).send().await.map_err(|e| format!("GET {url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("GET {url}: HTTP {}", resp.status()));
    }
    let val: Value = resp.json().await.map_err(|e| format!("GET {url}: bad JSON: {e}"))?;
    let ids: Vec<String> = val
        .get("data")
        .and_then(|d| d.as_array())
        .map(|a| a.iter().filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(String::from)).collect())
        .unwrap_or_default();
    match ids.len() {
        0 => Err(format!("{url} lists no models; pass --model")),
        1 => Ok(ids.into_iter().next().unwrap()),
        _ => Err(format!("{url} lists several models ({}); pass --model", ids.join(", "))),
    }
}

fn delta_has_text(chunk: &Value) -> bool {
    chunk
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("delta"))
        .map(|d| {
            ["content", "reasoning_content", "reasoning"]
                .iter()
                .any(|k| d.get(k).and_then(|v| v.as_str()).map(|s| !s.is_empty()).unwrap_or(false))
        })
        .unwrap_or(false)
}

async fn run_one(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    max_tokens: u32,
) -> Result<RunResult, String> {
    let body = json!({
        "model": model,
        "messages": [
            { "role": "system", "content": "You are a benchmark tester. Be brief." },
            { "role": "user", "content": "Explain how Mixture of Experts (MoE) gating routes tokens to top-k experts in 3 concise bullet points." }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": true,
        "stream_options": { "include_usage": true }
    });
    let start = Instant::now();
    let resp = client.post(url).json(&body).send().await.map_err(|e| format!("POST {url}: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        let snippet: String = text.chars().take(200).collect();
        return Err(format!("POST {url}: HTTP {status}: {snippet}"));
    }
    let mut ttft_ms: Option<f64> = None;
    let mut usage: Option<Value> = None;
    let mut timings: Option<Value> = None;
    let mut buf = String::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(|e| format!("stream error: {e}"))?;
        buf.push_str(&String::from_utf8_lossy(&bytes));
        while let Some(pos) = buf.find('\n') {
            let line: String = buf.drain(..=pos).collect();
            let Some(data) = line.trim().strip_prefix("data:") else { continue };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
            if ttft_ms.is_none() && delta_has_text(&v) {
                ttft_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
            }
            if v.get("usage").map(|u| u.is_object()).unwrap_or(false) {
                usage = v.get("usage").cloned();
            }
            if v.get("timings").map(|t| t.is_object()).unwrap_or(false) {
                timings = v.get("timings").cloned();
            }
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = ttft_ms.ok_or("stream carried no content delta")?;
    let usage = usage.ok_or("upstream returned no usage object in the stream")?;
    let completion_tokens = usage
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .ok_or("usage has no completion_tokens")?;
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .ok_or("usage has no prompt_tokens")?;
    let f = |k: &str| timings.as_ref().and_then(|t| t.get(k)).and_then(|v| v.as_f64());
    Ok(RunResult {
        ttft_ms,
        wall_ms,
        prompt_tokens,
        completion_tokens,
        upstream_predicted_ms: f("predicted_ms"),
        upstream_prompt_ms: f("prompt_ms"),
    })
}

pub async fn run_benchmark(
    upstream_url: &str,
    iterations: usize,
    model: Option<&str>,
    max_tokens: u32,
) -> Result<Scorecard, String> {
    if iterations == 0 {
        return Err("iterations must be at least 1".into());
    }
    let hw = crate::hardware::detect_hardware();
    let client = reqwest::Client::new();
    let model = resolve_model(&client, upstream_url, model).await?;
    let url = format!("{}/v1/chat/completions", upstream_url.trim_end_matches('/'));
    let mut runs = Vec::with_capacity(iterations);
    for i in 0..iterations {
        runs.push(run_one(&client, &url, &model, max_tokens).await.map_err(|e| format!("iteration {}: {e}", i + 1))?);
    }
    let runtime_metrics = aggregate(&runs, max_tokens)?;
    Ok(Scorecard {
        worker_id: format!("{}-lemonade-ori", hw.host_name),
        backend: "lemonade-ori-rust".into(),
        runtime: format!("upstream:{}", upstream_url),
        harness_host_gpus: hw.gpus.into_iter().map(|g| g.name).collect(),
        model,
        role: "lemonade_challenge_runner".into(),
        runtime_metrics,
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{http::StatusCode, response::IntoResponse, routing::{get, post}, Router};

    async fn serve(app: Router) -> String {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
        format!("http://{addr}")
    }

    const OK_SSE: &str = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\" there\"}}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20},",
        "\"timings\":{\"predicted_ms\":1000.0,\"prompt_ms\":100.0}}\n\n",
        "data: [DONE]\n\n"
    );

    fn models(ids: &'static [&'static str]) -> Router {
        Router::new().route(
            "/v1/models",
            get(move || async move {
                axum::Json(json!({"data": ids.iter().map(|i| json!({"id": i})).collect::<Vec<_>>()}))
            }),
        )
    }

    #[tokio::test]
    async fn success_uses_upstream_numbers() {
        let app = models(&["m1"]).route("/v1/chat/completions", post(|| async { OK_SSE }));
        let base = serve(app).await;
        let card = run_benchmark(&base, 2, None, 8).await.unwrap();
        let m = card.runtime_metrics;
        assert_eq!(card.model, "m1");
        assert_eq!(m.iterations, 2);
        assert_eq!(m.completion_tokens, 40);
        assert_eq!(m.prompt_tokens, 20);
        assert_eq!(m.decode_tps, 20.0);
        assert_eq!(m.decode_source, "upstream_timings");
        assert_eq!(m.prompt_tps, Some(100.0));
        assert!(m.mean_ttft_ms >= 0.0);
    }

    #[tokio::test]
    async fn upstream_500_fails() {
        let app = models(&["m1"]).route(
            "/v1/chat/completions",
            post(|| async { (StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response() }),
        );
        let base = serve(app).await;
        let err = run_benchmark(&base, 3, None, 8).await.unwrap_err();
        assert!(err.contains("500"), "{err}");
        assert!(err.contains("iteration 1"), "{err}");
    }

    #[tokio::test]
    async fn empty_usage_fails() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: [DONE]\n\n";
        let app = models(&["m1"]).route("/v1/chat/completions", post(move || async move { sse }));
        let base = serve(app).await;
        let err = run_benchmark(&base, 1, None, 8).await.unwrap_err();
        assert!(err.contains("no usage"), "{err}");
    }

    #[tokio::test]
    async fn missing_prompt_tokens_fails() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"completion_tokens\":20}}\n\n",
            "data: [DONE]\n\n"
        );
        let app = models(&["m1"]).route("/v1/chat/completions", post(move || async move { sse }));
        let base = serve(app).await;
        let err = run_benchmark(&base, 1, None, 8).await.unwrap_err();
        assert!(err.contains("prompt_tokens"), "{err}");
    }

    #[tokio::test]
    async fn multiple_models_require_explicit_choice() {
        let base = serve(models(&["a", "b"])).await;
        let err = run_benchmark(&base, 1, None, 8).await.unwrap_err();
        assert!(err.contains("pass --model"), "{err}");
    }

    #[test]
    fn aggregate_rejects_empty_and_zero_tokens() {
        assert!(aggregate(&[], 8).is_err());
        let r = RunResult { ttft_ms: 5.0, wall_ms: 10.0, prompt_tokens: 1, completion_tokens: 0, upstream_predicted_ms: None, upstream_prompt_ms: None };
        assert!(aggregate(&[r], 8).is_err());
    }

    #[test]
    fn aggregate_wall_clock_fallback() {
        let r = RunResult { ttft_ms: 500.0, wall_ms: 1500.0, prompt_tokens: 4, completion_tokens: 10, upstream_predicted_ms: None, upstream_prompt_ms: None };
        let m = aggregate(&[r], 8).unwrap();
        assert_eq!(m.decode_tps, 10.0);
        assert_eq!(m.decode_source, "wall_clock");
        assert_eq!(m.prompt_tps, None);
    }
}
