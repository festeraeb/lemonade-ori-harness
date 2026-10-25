//! Bearer tokens for the HTTP MCP endpoint and agent runs.
//!
//! Token file (JSON, mode 600): `{"operator": ["…"], "agent": ["…"]}`.
//! Operator tokens keep full shell/file access; agent tokens get the read-only policy.
//! With no token file, /mcp and /agent/* refuse every request.

use crate::policy::Tier;
use axum::http::HeaderMap;
use serde::Deserialize;

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Tokens {
    #[serde(default)]
    pub operator: Vec<String>,
    #[serde(default)]
    pub agent: Vec<String>,
    /// Action tier over MCP: a shell with everything but root.
    #[serde(default)]
    pub action: Vec<String>,
}

impl Tokens {
    pub fn load(path: &str) -> Option<Tokens> {
        let text = std::fs::read_to_string(path).ok()?;
        let t: Tokens = serde_json::from_str(&text).ok()?;
        // Short tokens are guessable; ignore them.
        let keep = |v: Vec<String>| v.into_iter().filter(|s| s.len() >= 24).collect::<Vec<_>>();
        let t = Tokens { operator: keep(t.operator), agent: keep(t.agent), action: keep(t.action) };
        (!t.operator.is_empty() || !t.agent.is_empty() || !t.action.is_empty()).then_some(t)
    }

    pub fn tier(&self, headers: &HeaderMap) -> Option<Tier> {
        let given = headers
            .get("authorization")
            .and_then(|h| h.to_str().ok())
            .and_then(|h| h.strip_prefix("Bearer "))?
            .trim();
        if self.operator.iter().any(|t| ct_eq(t, given)) {
            Some(Tier::Operator)
        } else if self.action.iter().any(|t| ct_eq(t, given)) {
            Some(Tier::Action)
        } else if self.agent.iter().any(|t| ct_eq(t, given)) {
            Some(Tier::Agent)
        } else {
            None
        }
    }
}

/// Compare without stopping at the first differing byte.
fn ct_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers_from_header() {
        let t = Tokens { operator: vec!["o".repeat(32)], agent: vec!["a".repeat(32)], action: vec![] };
        let mut h = HeaderMap::new();
        assert_eq!(t.tier(&h), None);
        h.insert("authorization", format!("Bearer {}", "a".repeat(32)).parse().unwrap());
        assert_eq!(t.tier(&h), Some(Tier::Agent));
        h.insert("authorization", format!("Bearer {}", "o".repeat(32)).parse().unwrap());
        assert_eq!(t.tier(&h), Some(Tier::Operator));
        h.insert("authorization", "Bearer nope".parse().unwrap());
        assert_eq!(t.tier(&h), None);
    }
}
