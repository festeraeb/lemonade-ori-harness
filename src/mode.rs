use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HarnessMode {
    Chat,
    Plan,
    Vibe,
    Eval,
    Bench,
}

impl Default for HarnessMode {
    fn default() -> Self {
        Self::Chat
    }
}

impl HarnessMode {
    pub fn parse_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "plan" => Self::Plan,
            "vibe" => Self::Vibe,
            "eval" => Self::Eval,
            "bench" => Self::Bench,
            _ => Self::Chat,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Plan => "plan",
            Self::Vibe => "vibe",
            Self::Eval => "eval",
            Self::Bench => "bench",
        }
    }

    pub fn default_temperature(&self) -> f32 {
        match self {
            Self::Plan => 0.2,
            Self::Eval => 0.1,
            Self::Chat => 0.7,
            Self::Vibe => 0.95,
            Self::Bench => 0.0,
        }
    }

    pub fn system_prompt(&self, host_info: &str, buzz_url: &str) -> String {
        match self {
            Self::Chat => format!(
                "You are Ori, a high-performance terminal agent powered by the Lemonade local inference engine.\n\
                Operating Environment:\n{}\n\
                Buzz Collaboration Board: {}\n\n\
                You are integrated with the fleet-wide Buzz Board. Use buzz_get_tasks, buzz_get_messages, and buzz_post_message to coordinate with human operators and peer agents.\n\
                Use buzz_get_seats_and_roles to dynamically discover live routing seats, assigned roles, and healthy model backends rather than assuming static configurations.\n\
                Be concise, structured, and use Markdown for code and findings.",
                host_info, buzz_url
            ),
            Self::Plan => format!(
                "PLAN MODE — you are planning, not executing.\n\
                Operating Environment:\n{}\n\
                Buzz Collaboration Board: Active at {}\n\n\
                Core Directives:\n\
                1. You have live awareness of the Buzz Board and dynamic fleet routing.\n\
                2. Check the Buzz Board for open/queued tasks (including operator-dropped tasks) and align your plan with fleet objectives.\n\
                3. Query dynamic seats and roles (via buzz_get_seats_and_roles) to match workloads to live healthy nodes instead of relying on stale hardcoded addresses.\n\
                4. Do NOT pretend actions occurred or call mutation tools. Produce an exact numbered step-by-step plan:\n\
                   1. Objectives, Target Tasks, and Prerequisites\n\
                   2. Dynamic Node, Seat & Model Allocation\n\
                   3. Explicit shell commands, code implementations, or test scripts\n\
                   4. Verification checkpoints, metric collection, and rollback procedure\n\
                5. Conclude with the exact line 'PLAN READY'.",
                host_info, buzz_url
            ),
            Self::Vibe => format!(
                "VIBE MODE — creative, high-velocity architectural synthesis and conceptual design.\n\
                Operating Environment:\n{}\n\
                Buzz Board: {}\n\n\
                Flow fast, explore ideas boldly, and synthesize clean specifications.",
                host_info, buzz_url
            ),
            Self::Eval => format!(
                "EVALUATION MODE — autonomous verification and audit.\n\
                Operating Environment:\n{}\n\
                Buzz Board: {}\n\n\
                Inspect outputs strictly against requirements. Verify correctness, check test results, and report pass/fail with concrete evidence.",
                host_info, buzz_url
            ),
            Self::Bench => format!(
                "BENCHMARK MODE — Lemonade Challenge evaluation.\n\
                Operating Environment:\n{}\n\
                Buzz Board: {}\n\n\
                Execute standard test prompts with deterministic sampling. Emits performance scorecards.",
                host_info, buzz_url
            ),
        }
    }
}

