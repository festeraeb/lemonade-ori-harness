mod agent;
mod auth;
mod bench;
mod hardware;
mod llm;
mod mcp;
mod mode;
mod policy;
mod site;
mod subagent;
mod ui;

use axum::{
    extract::State,
    http::HeaderMap,
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};

#[derive(Parser, Debug, Clone)]
#[command(
    name = "lemonade-ori",
    version = "0.1.0",
    about = "Standalone Ori Harness for Lemonade: flag-driven mode shifting, MCP surface layer, and Challenge scorecards."
)]
struct Args {
    #[arg(short, long, default_value = "8310", env = "LEMONADE_PORT")]
    port: u16,

    #[arg(long, default_value = "127.0.0.1", env = "LEMONADE_HOST",
        help = "Bind address. Default is loopback. /mcp and /agent/* need a bearer token from --token-file; operator tokens get full shell/file control.")]
    host: String,

    #[arg(long, default_value = "~/.config/ori/tokens.json", env = "ORI_TOKEN_FILE",
        help = "JSON {\"operator\": [...], \"agent\": [...]} (tokens of 24+ chars). Missing: /mcp and /agent/* refuse all requests.")]
    token_file: String,

    #[arg(long, default_value = "http://127.0.0.1:13305/api", env = "ORI_LEMONADE_URL",
        help = "Local Lemonade server for agent runs (upstream \"lemonade\")")]
    lemonade_url: String,

    #[arg(long, default_value = "Qwen3-VL-4B-Instruct-GGUF", env = "ORI_AGENT_MODEL",
        help = "Default model for agent runs and helpers")]
    agent_model: String,

    #[arg(long, default_value = "~/.config/ori/skills", env = "ORI_SKILLS_DIR",
        help = "Folder of <name>.md skill notes that rooms can ask for")]
    skills_dir: String,

    #[arg(long, default_value = "http://127.0.0.1:9135", env = "ORI_NIM_URL",
        help = "OpenAI-compatible NVIDIA NIM endpoints (or local proxies that add the API key), comma-separated, at most 5; upstream \"nim\" and spawn_subagents use them in turn")]
    nim_url: String,

    #[arg(long, default_value_t = 3, env = "ORI_MAX_AGENTS",
        help = "Subagents at once across the harness (ask_helper per run, spawn_subagents); capped at 5")]
    max_agents: usize,

    #[arg(long, default_value = "", env = "ORI_ACTION_MODELS",
        help = "Comma-separated models allowed the action tier besides NIM models and models over 2B")]
    action_models: String,

    #[arg(long, default_value = "~/.local/share/ori/runs", env = "ORI_RUNS_DIR", help = "One JSON transcript per agent run")]
    runs_dir: String,

    #[arg(long, default_value_t = 2, env = "ORI_MAX_RUNS", help = "Agent runs at once; more wait")]
    max_runs: usize,

    #[arg(short, long, default_value = "chat", env = "LEMONADE_MODE")]
    mode: String,

    #[arg(
        long,
        default_value = "http://127.0.0.1:8000",
        env = "LEMONADE_UPSTREAM"
    )]
    upstream: String,

    #[arg(
        long,
        default_value = "http://127.0.0.1:5560",
        env = "BUZZ_URL"
    )]
    buzz_url: String,

    #[arg(long, help = "Run benchmark suite immediately and exit")]
    bench: bool,

    #[arg(long, env = "LEMONADE_MODEL", help = "Model id for --bench (default: the single model listed by the upstream /v1/models; error if several)")]
    model: Option<String>,

    #[arg(long, default_value_t = 3, help = "Number of benchmark requests")]
    bench_iterations: usize,

    #[arg(long, default_value_t = 64, help = "max_tokens per benchmark request")]
    bench_max_tokens: u32,

    #[arg(long, help = "Run in MCP STDIO server mode for IDE integration")]
    mcp_stdio: bool,
}


#[derive(Clone)]
struct AppState {
    args: Args,
    hardware: hardware::HardwareProfile,
    client: reqwest::Client,
    runs: Arc<tokio::sync::Semaphore>,
    agents: Arc<tokio::sync::Semaphore>,
}

impl AppState {
    /// The caller's tier, re-reading the token file so a rotated token applies at once.
    fn tier(&self, headers: &HeaderMap) -> Result<policy::Tier, (StatusCode, Json<Value>)> {
        let Some(tokens) = auth::Tokens::load(&site::expand(&self.args.token_file)) else {
            return Err((StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": format!("no tokens configured in {}", self.args.token_file)}))));
        };
        tokens.tier(headers).ok_or((StatusCode::UNAUTHORIZED, Json(json!({"error": "missing or unknown bearer token"}))))
    }

    fn agent_config(&self) -> agent::Config {
        agent::Config {
            lemonade: self.args.lemonade_url.clone(),
            gateway: self.args.upstream.clone(),
            nim_urls: nim_urls(&self.args.nim_url),
            buzz: self.args.buzz_url.clone(),
            skills_dir: site::expand(&self.args.skills_dir),
            runs_dir: site::expand(&self.args.runs_dir),
            default_model: self.args.agent_model.clone(),
            action_models: self.args.action_models.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
            max_agents: max_agents(self.args.max_agents),
            agents_sem: self.agents.clone(),
        }
    }
}

/// The NIM pool from `--nim-url`: comma-separated, blanks dropped, at most 5.
fn nim_urls(arg: &str) -> Vec<String> {
    arg.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).take(5).collect()
}

/// `--max-agents`, kept between 1 and 5.
fn max_agents(n: usize) -> usize {
    n.clamp(1, 5)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    // 1. If --bench flag is given, run benchmark immediately and exit
    if args.bench {
        println!("🚀 Running Lemonade Challenge Benchmark Suite...");
        match bench::run_benchmark(&args.upstream, args.bench_iterations, args.model.as_deref(), args.bench_max_tokens).await {
            Ok(card) => {
                println!("{}", serde_json::to_string_pretty(&card)?);
                return Ok(());
            }
            Err(e) => {
                eprintln!("Benchmark failed: {}", e);
                std::process::exit(1);
            }
        }
    }

    // 2. If --mcp-stdio is given, run as standard MCP STDIO server
    if args.mcp_stdio {
        let env = mcp::HarnessEnv {
            gateway: args.upstream.clone(),
            nim_urls: nim_urls(&args.nim_url),
            buzz: args.buzz_url.clone(),
            max_agents: max_agents(args.max_agents),
            agents_sem: Arc::new(tokio::sync::Semaphore::new(max_agents(args.max_agents))),
        };
        return run_stdio_mcp(&env).await;
    }

    // 3. Normal Daemon Mode: Probe hardware and launch HTTP / MCP server
    let hw = hardware::detect_hardware();
    info!(
        "Lemonade Ori starting on host '{}' ({} CPU cores, {:.1} GB RAM)",
        hw.host_name, hw.cpu_cores, hw.total_memory_gb
    );
    for gpu in &hw.gpus {
        info!("Detected acceleration: {} [{}]", gpu.name, gpu.backend);
    }
    info!("Default mode: {}", args.mode);
    info!("Upstream endpoint: {}", args.upstream);

    let state = Arc::new(AppState {
        args: args.clone(),
        hardware: hw,
        client: reqwest::Client::new(),
        runs: Arc::new(tokio::sync::Semaphore::new(args.max_runs.max(1))),
        agents: Arc::new(tokio::sync::Semaphore::new(max_agents(args.max_agents))),
    });
    if auth::Tokens::load(&site::expand(&args.token_file)).is_none() {
        warn!("no tokens in {}: /mcp and /agent/* will refuse every request", args.token_file);
    }

    let app = Router::new()
        .route("/", get(handle_ui))
        .route("/warp", get(handle_ui))
        .route("/health", get(handle_health))
        .route("/mcp", post(handle_mcp))
        .route("/agent/run", post(handle_agent_run))
        .route("/agent/runs", get(handle_agent_runs))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/models", get(handle_models))
        .route("/models", get(handle_models))
        .route("/api/mode", get(handle_get_mode))
        .route("/api/bench", post(handle_bench_api))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    if !addr.ip().is_loopback() {
        warn!("==================================================================");
        warn!("WARNING: bound to non-loopback address {}.", addr);
        warn!("POST /mcp with an operator token gives full shell and file control of this machine.");
        warn!("Keep the token file private; prefer 127.0.0.1 (the default).");
        warn!("==================================================================");
    }
    info!("Lemonade Ori Harness listening on http://{}", addr);
    info!("Web UI available at http://{}/", addr);
    info!("MCP endpoint available at http://{}/mcp", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn handle_ui(State(state): State<Arc<AppState>>) -> Html<String> {
    Html(ui::render_html(
        &state.hardware.host_name,
        &state.hardware.recommended_backend,
    ))
}

async fn handle_health(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "harness": "lemonade-ori-rust",
        "host": state.hardware.host_name,
        "default_mode": state.args.mode,
        "acceleration": state.hardware.recommended_backend,
        "gpus": state.hardware.gpus,
        "upstream": state.args.upstream,
        // Agent runs going on now; wait for 0 before restarting the service.
        "runs_active": state.args.max_runs.max(1) - state.runs.available_permits()
    }))
}

async fn handle_get_mode(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "mode": state.args.mode,
        "supported_modes": ["chat", "plan", "vibe", "eval", "bench"]
    }))
}

async fn handle_mcp(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(rpc_req): Json<mcp::JsonRpcRequest>,
) -> Response {
    let tier = match state.tier(&headers) {
        Ok(t) => t,
        Err(e) => return e.into_response(),
    };
    let resp = mcp::handle_json_rpc(rpc_req, &state.agent_config().harness_env(), &state.client, tier).await;
    Json(resp).into_response()
}

/// Run one unattended agent turn (the waker calls this). Always runs with the agent policy,
/// whichever token started it.
async fn handle_agent_run(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<agent::RunRequest>,
) -> Response {
    if let Err(e) = state.tier(&headers) {
        return e.into_response();
    }
    let Ok(_permit) = state.runs.clone().acquire_owned().await else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "shutting down"}))).into_response();
    };
    info!("agent run: {} in #{}: {}", req.agent, req.room, req.goal.chars().take(120).collect::<String>());
    match agent::run(&state.agent_config(), &state.client, req).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => {
            warn!("agent run failed: {e}");
            (StatusCode::BAD_GATEWAY, Json(json!({"status": "error", "error": e}))).into_response()
        }
    }
}

/// The latest run transcripts' summaries, newest first.
async fn handle_agent_runs(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(e) = state.tier(&headers) {
        return e.into_response();
    }
    let mut files: Vec<_> = std::fs::read_dir(&state.args.runs_dir)
        .map(|d| d.filter_map(|e| e.ok()).map(|e| e.path()).collect())
        .unwrap_or_default();
    files.sort();
    files.reverse();
    let runs: Vec<Value> = files
        .iter()
        .take(50)
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .filter_map(|t| serde_json::from_str::<Value>(&t).ok())
        .map(|v| json!({
            "run_id": v["run_id"], "agent": v["agent"], "room": v["room"], "status": v["status"],
            "model": v["model"], "posted_id": v["posted_id"],
            "steps": v["steps"].as_array().map(|a| a.len()).unwrap_or(0),
            "goal": v["goal"],
        }))
        .collect();
    Json(json!({ "runs": runs })).into_response()
}

async fn handle_bench_api(State(state): State<Arc<AppState>>) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    match bench::run_benchmark(&state.args.upstream, state.args.bench_iterations, state.args.model.as_deref(), state.args.bench_max_tokens).await {
        Ok(card) => Ok(Json(serde_json::to_value(card).unwrap_or_default())),
        Err(e) => {
            warn!("benchmark failed: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))))
        }
    }
}

async fn handle_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut model_list = vec![
        json!({
            "id": "lemonade-ori",
            "object": "model",
            "owned_by": "lemonade-rust-harness",
            "node": state.hardware.host_name,
            "acceleration": state.hardware.recommended_backend
        }),
        json!({
            "id": "lemonade-challenge",
            "object": "model",
            "owned_by": "lemonade-rust-harness",
            "node": state.hardware.host_name
        }),
        json!({
            "id": "lemonade-local",
            "object": "model",
            "owned_by": "lemonade-rust-harness",
            "node": state.hardware.host_name
        }),
    ];

    let url = format!("{}/v1/models", state.args.upstream.trim_end_matches('/'));
    if let Ok(resp) = state.client.get(&url).send().await {
        if let Ok(val) = resp.json::<Value>().await {
            if let Some(upstream_arr) = val.get("data").and_then(|d| d.as_array()) {
                for m in upstream_arr {
                    model_list.push(m.clone());
                }
            }
        }
    }

    Json(json!({
        "object": "list",
        "data": model_list
    }))
}

async fn resolve_dynamic_model(state: &AppState, active_mode: &mode::HarnessMode) -> String {
    // 1. Probe live dynamic routing table at {upstream}/routing
    let routing_url = format!("{}/routing", state.args.upstream.trim_end_matches('/'));
    if let Ok(resp) = state.client.get(&routing_url).send().await {
        if let Ok(val) = resp.json::<Value>().await {
            if let Some(roles) = val.get("paddler").and_then(|p| p.get("roles")) {
                let preferred_role = match active_mode {
                    mode::HarnessMode::Plan => "planner",
                    _ => "coder",
                };
                if let Some(chosen) = roles.get(preferred_role).and_then(|r| r.get("chosen")).and_then(|c| c.as_str()) {
                    if chosen != "auto" && chosen != "lemonade-ori" && chosen != "paddler" && chosen != "strata-c4" {
                        return chosen.to_string();
                    }
                }
                if let Some(chosen) = roles.get("auto").and_then(|r| r.get("chosen")).and_then(|c| c.as_str()) {
                    if chosen != "auto" && chosen != "lemonade-ori" && chosen != "paddler" && chosen != "strata-c4" {
                        return chosen.to_string();
                    }
                }
            }
            if let Some(seats) = val.get("seats").and_then(|s| s.as_array()) {
                for s in seats {
                    if s.get("healthy").and_then(|h| h.as_bool()).unwrap_or(false) {
                        if let Some(m) = s.get("model_id").and_then(|id| id.as_str()) {
                            if m != "strata-c4" {
                                return m.to_string();
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Query /v1/models for reachable models
    let models_url = format!("{}/v1/models", state.args.upstream.trim_end_matches('/'));
    if let Ok(resp) = state.client.get(&models_url).send().await {
        if let Ok(val) = resp.json::<Value>().await {
            if let Some(arr) = val.get("data").and_then(|d| d.as_array()) {
                let candidates = match active_mode {
                    mode::HarnessMode::Plan => &["deepseek-v4-flash-9b", "glm-5.3-flash", "katcoder-35b-a3b", "katcoder"],
                    _ => &["deepseek-v4-flash-9b", "glm-5.3-flash", "katcoder-35b-a3b", "katcoder"],
                };
                for candidate in candidates {
                    if arr.iter().any(|m| m.get("id").and_then(|id| id.as_str()) == Some(*candidate)) {
                        return candidate.to_string();
                    }
                }
            }
        }
    }

    // Default live seat fallback
    "deepseek".to_string()
}

async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Response {
    // Mode shifting: check header first, then body, then CLI default
    let mode_str = headers
        .get("x-lemonade-mode")
        .and_then(|h| h.to_str().ok())
        .or_else(|| body.get("mode").and_then(|m| m.as_str()))
        .unwrap_or(&state.args.mode);

    let active_mode = mode::HarnessMode::parse_str(mode_str);

    // Rewrite model if lemonade-ori or generic to avoid upstream recursion loops
    if let Some(map) = body.as_object_mut() {
        let needs_rewrite = match map.get("model").and_then(|v| v.as_str()) {
            Some(m) => {
                m == "lemonade-ori"
                    || m == "lemonade-challenge"
                    || m == "lemonade-local"
                    || m == "auto"
                    || m == "default"
                    || m == "paddler"
                    || m == "strata-c4"
            }
            None => true,
        };
        if needs_rewrite {
            let dynamic_model = resolve_dynamic_model(&state, &active_mode).await;
            map.insert("model".into(), json!(dynamic_model));
        }
        // Set preset to "local" and from_lemonade to avoid recursive loopback in gateway
        map.insert("preset".into(), json!("local"));
        map.insert("from_lemonade".into(), json!(true));
    }


    // Apply temperature if not explicitly supplied
    if body.get("temperature").is_none() {
        if let Some(map) = body.as_object_mut() {
            map.insert(
                "temperature".into(),
                json!(active_mode.default_temperature()),
            );
        }
    }

    // Prepend mode system prompt
    let host_info = format!(
        "- Node: {}\n- Hardware: {}\n- Backend: {}",
        state.hardware.host_name,
        state
            .hardware
            .gpus
            .first()
            .map(|g| g.name.as_str())
            .unwrap_or("CPU"),
        state.hardware.recommended_backend
    );

    let sys_prompt = active_mode.system_prompt(&host_info, &state.args.buzz_url);

    if let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        let has_sys = messages
            .first()
            .and_then(|m| m.get("role"))
            .and_then(|r| r.as_str())
            == Some("system");
        if !has_sys {
            messages.insert(
                0,
                json!({
                    "role": "system",
                    "content": sys_prompt
                }),
            );
        }
    }

    // Proxy to upstream Lemonade endpoint
    let url = format!(
        "{}/v1/chat/completions",
        state.args.upstream.trim_end_matches('/')
    );
    match state.client.post(&url).json(&body).send().await {
        Ok(upstream_resp) => {
            let status = StatusCode::from_u16(upstream_resp.status().as_u16())
                .unwrap_or(StatusCode::OK);
            let bytes = upstream_resp.bytes().await.unwrap_or_default();
            (status, bytes).into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": { "message": format!("Upstream connection failed: {}", e) } })),
        )
            .into_response(),
    }
}

async fn run_stdio_mcp(env: &mcp::HarnessEnv) -> anyhow::Result<()> {
    use std::io::{self, BufRead};
    let stdin = io::stdin();
    let client = reqwest::Client::new();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(rpc_req) = serde_json::from_str::<mcp::JsonRpcRequest>(&line) {
            // stdio: the person who started this process already has their own shell.
            let resp = mcp::handle_json_rpc(rpc_req, env, &client, policy::Tier::Operator).await;
            println!("{}", serde_json::to_string(&resp)?);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nim_pool_and_agent_cap() {
        assert_eq!(nim_urls(" http://a , ,http://b"), vec!["http://a", "http://b"]);
        assert_eq!(nim_urls("1,2,3,4,5,6,7").len(), 5);
        assert!(nim_urls("").is_empty());
        assert_eq!(max_agents(0), 1);
        assert_eq!(max_agents(9), 5);
    }
}
