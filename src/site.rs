//! Facts about the machine Ori runs on, read once from the environment, so no host names,
//! addresses or paths are compiled in.
//!
//! - `ORI_HOST_NAME`: how prompts name this machine (default: the system host name).
//! - `ORI_SELF_HOSTS`: extra names and addresses of this machine, comma-separated. Agents may not
//!   ssh to any of them, because that would leave the no-root sandbox.
//! - `ORI_WORK_DIR`: where action agents make their git worktrees and run commands (default `~/ori-work`).

use std::sync::OnceLock;

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

pub fn home() -> String {
    env("HOME").or_else(|| env("USERPROFILE")).unwrap_or_else(|| ".".into())
}

/// `~/x` becomes `$HOME/x`; anything else is returned unchanged.
pub fn expand(path: &str) -> String {
    match path.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", home().trim_end_matches('/')),
        None if path == "~" => home(),
        None => path.to_string(),
    }
}

pub fn host_name() -> &'static str {
    static NAME: OnceLock<String> = OnceLock::new();
    NAME.get_or_init(|| {
        env("ORI_HOST_NAME")
            .or_else(|| std::fs::read_to_string("/proc/sys/kernel/hostname").ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()))
            .or_else(|| env("COMPUTERNAME"))
            .or_else(|| env("HOSTNAME"))
            .unwrap_or_else(|| "this machine".into())
    })
}

pub fn user_name() -> String {
    env("USER").or_else(|| env("USERNAME")).unwrap_or_else(|| "the service user".into())
}

pub fn work_dir() -> &'static str {
    static DIR: OnceLock<String> = OnceLock::new();
    DIR.get_or_init(|| expand(&env("ORI_WORK_DIR").unwrap_or_else(|| "~/ori-work".into())))
}

/// Lower-case names and address prefixes that mean "this machine".
pub fn self_hosts() -> &'static [String] {
    static HOSTS: OnceLock<Vec<String>> = OnceLock::new();
    HOSTS.get_or_init(|| {
        let mut v: Vec<String> = ["localhost", "127.", "::1"].iter().map(|s| s.to_string()).collect();
        if host_name() != "this machine" {
            v.push(host_name().to_lowercase());
        }
        if let Some(extra) = env("ORI_SELF_HOSTS") {
            v.extend(extra.split(',').map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()));
        }
        v
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_home() {
        let h = home();
        assert_eq!(expand("~/a/b"), format!("{}/a/b", h.trim_end_matches('/')));
        assert_eq!(expand("/abs"), "/abs");
        assert!(self_hosts().iter().any(|h| h == "localhost"));
    }
}
