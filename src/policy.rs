//! What an unattended agent may run. Agents get read-only commands only (the "green" tier):
//! no shell, so no redirection, expansion or chaining; a fixed list of programs, each with its
//! arguments checked; secrets unreadable. Writes go through the forge as Buzz tasks, never here.
//! The operator token keeps the original unrestricted `shell_exec`.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Full control, as before: only for tokens listed under "operator".
    Operator,
    /// Research tier: read, measure and post. Small models and plain questions get this.
    Agent,
    /// Action tier: a real shell as the service user, with root shut off (no_new_privs, so sudo cannot
    /// elevate) and root-equivalent tools refused. NIM models and models over 2B parameters.
    Action,
}

/// Models at or below this many billion parameters only get the research tier.
pub const ACTION_MIN_PARAMS_B: f32 = 2.0;

/// The largest "<number>B" in a model id, in billions: "tiel-coder-35b-a3b" -> 35,
/// "Qwen3-VL-4B-Instruct" -> 4, "Qwen3-0.6B" -> 0.6, "bitnet-b1.58-2B-4T" -> 2. None if absent.
pub fn params_b(model: &str) -> Option<f32> {
    let b = model.as_bytes();
    let mut best: Option<f32> = None;
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() && (i == 0 || !b[i - 1].is_ascii_digit() && b[i - 1] != b'.') {
            let start = i;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                i += 1;
            }
            let after_ok = i + 1 >= b.len() || !b[i + 1].is_ascii_alphabetic();
            if i < b.len() && (b[i] == b'b' || b[i] == b'B') && after_ok {
                if let Ok(v) = model[start..i].parse::<f32>() {
                    best = Some(best.map_or(v, |x: f32| x.max(v)));
                }
            }
        } else {
            i += 1;
        }
    }
    best
}

/// NIM models (vendor/model ids), models over 2B, and any model the operator lists.
pub fn action_allowed(model: &str, extra: &[String]) -> bool {
    model.contains('/')
        || params_b(model).map(|p| p > ACTION_MIN_PARAMS_B).unwrap_or(false)
        || extra.iter().any(|m| m == model)
}

/// Commands that become root, or tools whose groups are root-equivalent (docker, lxd, libvirt).
const ROOT_WORDS: &[&str] = &[
    "sudo", "su", "doas", "pkexec", "run0", "setpriv", "nsenter", "machinectl", "systemd-run", "chroot",
    "docker", "podman", "lxc", "lxd", "incus", "virsh", "virt-install", "virt-manager",
];
/// Ori's own token file and private keys stay out of reach (a stolen operator token would be a way around the tier).
const ACTION_SECRETS: &[&str] = &["/.config/ori", "tokens.json", "/.ssh/id_", "credentials.sh", "sudoers", "/etc/shadow"];

pub fn check_action_command(cmd: &str) -> Result<(), String> {
    if cmd.trim().is_empty() {
        return Err("cmd is required".into());
    }
    let words: Vec<&str> = cmd
        .split(|c: char| c.is_whitespace() || ";|&()`<>'\"=$".contains(c))
        .filter(|w| !w.is_empty())
        .map(|w| w.rsplit('/').next().unwrap_or(w))
        .collect();
    if let Some(w) = words.iter().find(|w| ROOT_WORDS.contains(w)) {
        return Err(format!("{w}: root and root-equivalent tools are off limits for agents (ask the operator in #help)"));
    }
    let lower = cmd.to_lowercase();
    if let Some(s) = ACTION_SECRETS.iter().find(|s| lower.contains(*s)) {
        return Err(format!("{s}: secrets and Ori's own credentials are off limits"));
    }
    if words.iter().any(|w| matches!(*w, "ssh" | "scp" | "rsync" | "sftp")) {
        // Logging back into this machine would start a shell without no_new_privs.
        if let Some(h) = crate::site::self_hosts().iter().find(|h| lower.contains(h.as_str())) {
            return Err(format!("ssh to this machine ({h}) is not allowed: it would leave the no-root sandbox"));
        }
    }
    Ok(())
}

pub fn check_action_path(path: &str) -> Result<(), String> {
    let lower = std::fs::canonicalize(path)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string())
        .to_lowercase();
    match ACTION_SECRETS.iter().find(|s| lower.contains(*s)) {
        Some(s) => Err(format!("{path}: off limits ({s})")),
        None => Ok(()),
    }
}

/// Run `cmd` with bash under no_new_privs (sudo and other setuid programs cannot raise privileges).
pub fn run_action(cmd: &str, cwd: &str, timeout: Duration) -> Result<(String, i32), String> {
    let _ = std::fs::create_dir_all(cwd);
    let mut child = Command::new("setpriv")
        .args(["--no-new-privs", "--", "bash", "-c", cmd])
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("setpriv: {e}"))?;
    // Read both pipes on threads so a chatty command cannot block on a full pipe.
    let mut so = child.stdout.take().unwrap();
    let mut se = child.stderr.take().unwrap();
    let t_out = std::thread::spawn(move || {
        let mut v = Vec::new();
        let _ = so.by_ref().take(MAX_OUTPUT as u64).read_to_end(&mut v);
        let _ = std::io::copy(&mut so, &mut std::io::sink());
        v
    });
    let t_err = std::thread::spawn(move || {
        let mut v = Vec::new();
        let _ = se.by_ref().take(16 * 1024).read_to_end(&mut v);
        let _ = std::io::copy(&mut se, &mut std::io::sink());
        v
    });
    let start = Instant::now();
    let code = loop {
        if let Some(st) = child.try_wait().map_err(|e| e.to_string())? {
            break st.code().unwrap_or(-1);
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("timed out after {}s", timeout.as_secs()));
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    // A background process started by the command keeps the pipes open; do not wait on it forever.
    let grace = Instant::now();
    while !(t_out.is_finished() && t_err.is_finished()) && grace.elapsed() < Duration::from_secs(3) {
        std::thread::sleep(Duration::from_millis(50));
    }
    if !(t_out.is_finished() && t_err.is_finished()) {
        return Ok((format!("(exit {code}; output not collected: a background process still holds stdout/stderr)"), code));
    }
    let mut out = String::from_utf8_lossy(&t_out.join().unwrap_or_default()).to_string();
    let err = String::from_utf8_lossy(&t_err.join().unwrap_or_default()).to_string();
    if !err.trim().is_empty() {
        out.push_str("\n[stderr]: ");
        out.push_str(err.trim());
    }
    Ok((out, code))
}

pub const MAX_OUTPUT: usize = 64 * 1024;

/// Substrings that mark a path as secret; matched against the lowercased, resolved path.
const SECRET_MARKERS: &[&str] = &[
    "/.ssh", "/.gnupg", "/.config/", "/.claude", "/.codex", "/.copilot", "/.aws", "/.netrc", ".git-credentials",
    "secret", "token", "credential", "password", "passwd", "shadow", "sudoers", ".env", "id_rsa", "id_ed25519",
    "apikey", "api_key", "api-key", ".key", ".pem",
];

/// Refuse paths that look like secrets. The path is resolved first, so a symlink cannot hide one.
pub fn check_read_path(path: &str) -> Result<PathBuf, String> {
    if path.trim().is_empty() {
        return Err("path is required".into());
    }
    let resolved = std::fs::canonicalize(path).map_err(|e| format!("{path}: {e}"))?;
    let lower = resolved.to_string_lossy().to_lowercase();
    if let Some(m) = SECRET_MARKERS.iter().find(|m| lower.contains(*m)) {
        return Err(format!("{path}: refused, looks like a secret (matches {m:?})"));
    }
    Ok(resolved)
}

/// Split a command line into pipeline stages of argv. Quotes group words; `|` outside quotes
/// separates stages; anything a shell would interpret (`;&<>$()` and backticks, newlines)
/// outside quotes is refused, because nothing here runs a shell.
pub fn parse_pipeline(cmd: &str) -> Result<Vec<Vec<String>>, String> {
    let mut stages: Vec<Vec<String>> = vec![Vec::new()];
    let mut word = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    let mut chars = cmd.chars().peekable();
    while let Some(c) = chars.next() {
        match quote {
            Some(q) if c == q => quote = None,
            Some('"') if c == '\\' => {
                if let Some(n) = chars.next() {
                    word.push(n);
                }
            }
            Some(_) => word.push(c),
            None => match c {
                '\'' | '"' => {
                    quote = Some(c);
                    in_word = true;
                }
                '\\' => {
                    if let Some(n) = chars.next() {
                        word.push(n);
                        in_word = true;
                    }
                }
                '|' => {
                    if in_word {
                        stages.last_mut().unwrap().push(std::mem::take(&mut word));
                        in_word = false;
                    }
                    stages.push(Vec::new());
                }
                ';' | '&' | '<' | '>' | '$' | '`' | '(' | ')' | '\n' | '\r' => {
                    return Err(format!("{c:?} is not allowed: commands run without a shell"));
                }
                c if c.is_whitespace() => {
                    if in_word {
                        stages.last_mut().unwrap().push(std::mem::take(&mut word));
                        in_word = false;
                    }
                }
                _ => {
                    word.push(c);
                    in_word = true;
                }
            },
        }
    }
    if quote.is_some() {
        return Err("unclosed quote".into());
    }
    if in_word {
        stages.last_mut().unwrap().push(word);
    }
    if stages.iter().any(|s| s.is_empty()) {
        return Err("empty command or pipeline stage".into());
    }
    if stages.len() > 4 {
        return Err("at most 4 pipeline stages".into());
    }
    Ok(stages)
}

fn deny_flags(args: &[String], bad: &[&str]) -> Result<(), String> {
    for a in args {
        if let Some(b) = bad.iter().find(|b| a == *b || a.starts_with(&format!("{b}="))) {
            return Err(format!("flag {b} is not allowed for agents"));
        }
    }
    Ok(())
}

/// Path-looking arguments must pass the secret check.
fn check_paths(args: &[String]) -> Result<(), String> {
    for a in args.iter().filter(|a| a.starts_with('/') || a.starts_with('~') || a.contains("..")) {
        if a.starts_with('~') || a.contains("..") {
            return Err(format!("{a}: use an absolute path without ~ or .."));
        }
        if Path::new(a).exists() {
            check_read_path(a)?;
        } else {
            let lower = a.to_lowercase();
            if let Some(m) = SECRET_MARKERS.iter().find(|m| lower.contains(*m)) {
                return Err(format!("{a}: refused, looks like a secret (matches {m:?})"));
            }
        }
    }
    Ok(())
}

/// The allowlist: program name -> argument check. Every program here only reads.
pub fn check_stage(argv: &[String]) -> Result<(), String> {
    let prog = argv[0].as_str();
    let args = &argv[1..];
    if prog.contains('/') {
        return Err(format!("{prog}: give the program name, not a path"));
    }
    match prog {
        // Plain readers and filters.
        "cat" | "head" | "tail" | "wc" | "grep" | "ls" | "stat" | "sort" | "uniq" | "cut" | "tr" | "column"
        | "du" | "df" | "lsblk" | "file" | "md5sum" | "sha256sum" => check_paths(args),
        "free" | "uptime" | "uname" | "nproc" | "lscpu" | "lspci" | "sensors" | "date" | "whoami" | "id" | "ss"
        | "ps" | "vmstat" | "iostat" | "mpstat" | "lsmod" => check_paths(args),
        "hostname" => {
            if args.iter().any(|a| !a.starts_with('-')) {
                return Err("hostname with an argument would set the name".into());
            }
            Ok(())
        }
        "find" => {
            deny_flags(args, &["-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"])?;
            check_paths(args)
        }
        "nvidia-smi" => {
            // Queries only: no clocks, power limits, persistence, resets or modes.
            let ok = |a: &String| {
                a.starts_with("--query") || a.starts_with("--format") || a.starts_with("-i") || a.starts_with("--id")
                    || matches!(a.as_str(), "-q" | "-L" | "--list-gpus" | "topo" | "-m" | "nvlink" | "-s" | "-d" | "-x")
                    || !a.starts_with('-') && args.iter().any(|p| p == "-d" || p == "-i" || p == "--id")
            };
            if let Some(bad) = args.iter().find(|a| !ok(a)) {
                return Err(format!("nvidia-smi {bad}: only queries are allowed"));
            }
            Ok(())
        }
        "systemctl" => match args.iter().find(|a| !a.starts_with('-')).map(|s| s.as_str()) {
            Some("status" | "is-active" | "is-failed" | "is-enabled" | "list-units" | "list-timers" | "show" | "cat"
            | "list-unit-files" | "--failed") | None => Ok(()),
            Some(other) => Err(format!("systemctl {other}: only status-type subcommands are allowed")),
        },
        "journalctl" => deny_flags(args, &["--vacuum-size", "--vacuum-time", "--vacuum-files", "--rotate", "--flush", "--sync", "--relinquish-var", "--smart-relinquish-var", "--setup-keys", "--update-catalog"]),
        "ip" => {
            if args.iter().any(|a| matches!(a.as_str(), "add" | "del" | "delete" | "set" | "change" | "replace" | "flush" | "append" | "prepend")) {
                return Err("ip: only show/list is allowed".into());
            }
            Ok(())
        }
        "git" => {
            let sub = args.iter().find(|a| !a.starts_with('-')).map(|s| s.as_str()).unwrap_or("");
            match sub {
                "status" | "log" | "diff" | "show" | "blame" | "shortlog" | "rev-parse" | "ls-files" | "grep" => {
                    deny_flags(args, &["--output", "-o"])?;
                    check_paths(args)
                }
                "branch" | "tag" | "remote" => {
                    // Listing only: any positional after the subcommand would create or delete.
                    let pos = args.iter().filter(|a| !a.starts_with('-')).count();
                    if pos > 1 || args.iter().any(|a| matches!(a.as_str(), "-d" | "-D" | "-m" | "-M" | "-c" | "-C" | "--delete" | "add" | "remove")) {
                        return Err(format!("git {sub}: listing only"));
                    }
                    Ok(())
                }
                _ => Err(format!("git {sub}: only read-only git subcommands are allowed")),
            }
        }
        "curl" => {
            deny_flags(args, &["-X", "--request", "-d", "--data", "--data-raw", "--data-binary", "--data-urlencode", "-F", "--form", "-T", "--upload-file", "-o", "--output", "-O", "--remote-name", "-K", "--config", "-u", "--user", "-b", "--cookie", "-c", "--cookie-jar", "--json", "--next", "-:"])?;
            // Bundled short flags (`-sXPOST`) hide the same options.
            if let Some(a) = args.iter().find(|a| a.starts_with('-') && !a.starts_with("--") && a[1..].chars().any(|c| "XdFToOKubc:".contains(c))) {
                return Err(format!("curl {a}: writes, uploads and custom methods are not allowed"));
            }
            // Every URL must be a fleet address; skip the values of flags that take one.
            const TAKES_VALUE: &[&str] = &["-H", "--header", "-m", "--max-time", "--connect-timeout", "-w", "--write-out", "-A", "--user-agent", "--retry"];
            let mut skip = false;
            for a in args {
                if skip {
                    skip = false;
                    continue;
                }
                if a.starts_with('-') {
                    skip = TAKES_VALUE.contains(&a.as_str());
                    continue;
                }
                let host = a.split("://").last().unwrap_or(a).split(['/', ':', '?']).next().unwrap_or("");
                let fleet = host == "localhost" || host.starts_with("127.") || host.starts_with("10.") || host.starts_with("192.168.") || host.starts_with("100.");
                if !fleet {
                    return Err(format!("curl {host}: only local and private network addresses (127.*, 10.*, 192.168.*, 100.*)"));
                }
            }
            Ok(())
        }
        "docker" => match args.first().map(|s| s.as_str()) {
            Some("ps" | "images" | "logs" | "inspect" | "version" | "info") => Ok(()),
            Some("stats") if args.iter().any(|a| a == "--no-stream") => Ok(()),
            other => Err(format!("docker {}: read-only subcommands only", other.unwrap_or(""))),
        },
        _ => Err(format!("{prog}: not on the agent allowlist (read-only fleet commands only)")),
    }
}

pub fn check_command(cmd: &str) -> Result<Vec<Vec<String>>, String> {
    let stages = parse_pipeline(cmd)?;
    for s in &stages {
        check_stage(s)?;
    }
    Ok(stages)
}

/// Run a checked pipeline without a shell; kill it all after `timeout`. Returns (output, exit code).
pub fn run_pipeline(stages: &[Vec<String>], timeout: Duration) -> Result<(String, i32), String> {
    let mut children: Vec<Child> = Vec::new();
    let mut prev_stdout: Option<std::process::ChildStdout> = None;
    for (i, argv) in stages.iter().enumerate() {
        let mut c = Command::new(&argv[0]);
        c.args(&argv[1..]).stderr(Stdio::piped()).stdout(Stdio::piped());
        match prev_stdout.take() {
            Some(out) => c.stdin(Stdio::from(out)),
            None => c.stdin(Stdio::null()),
        };
        let mut child = c.spawn().map_err(|e| {
            for ch in children.iter_mut() {
                let _ = ch.kill();
            }
            format!("{}: {e}", argv[0])
        })?;
        if i + 1 < stages.len() {
            prev_stdout = child.stdout.take();
        }
        children.push(child);
    }
    let start = Instant::now();
    let code = loop {
        let last = children.last_mut().unwrap();
        if let Some(status) = last.try_wait().map_err(|e| e.to_string())? {
            break status.code().unwrap_or(-1);
        }
        if start.elapsed() > timeout {
            for ch in children.iter_mut() {
                let _ = ch.kill();
            }
            return Err(format!("timed out after {}s", timeout.as_secs()));
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let mut out = String::new();
    let last = children.last_mut().unwrap();
    if let Some(mut s) = last.stdout.take() {
        let mut buf = Vec::new();
        let _ = s.by_ref().take(MAX_OUTPUT as u64).read_to_end(&mut buf);
        out.push_str(&String::from_utf8_lossy(&buf));
    }
    let mut err = String::new();
    for ch in children.iter_mut() {
        if let Some(mut s) = ch.stderr.take() {
            let mut buf = Vec::new();
            let _ = s.by_ref().take(8 * 1024).read_to_end(&mut buf);
            err.push_str(&String::from_utf8_lossy(&buf));
        }
        let _ = ch.kill();
        let _ = ch.wait();
    }
    if !err.trim().is_empty() {
        out.push_str("\n[stderr]: ");
        out.push_str(err.trim());
    }
    Ok((out, code))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_quotes_and_pipes() {
        let s = parse_pipeline("grep -n 'a | b' /etc/hostname | head -3").unwrap();
        assert_eq!(s, vec![vec!["grep", "-n", "a | b", "/etc/hostname"], vec!["head", "-3"]]);
        assert!(parse_pipeline("ls; rm -rf /").is_err());
        assert!(parse_pipeline("cat x > y").is_err());
        assert!(parse_pipeline("echo $(id)").is_err());
        assert!(parse_pipeline("ls 'unclosed").is_err());
        assert!(parse_pipeline("ls |").is_err());
    }

    #[test]
    fn allowlist() {
        assert!(check_command("df -h / | tail -1").is_ok());
        assert!(check_command("nvidia-smi --query-gpu=name,temperature.gpu --format=csv").is_ok());
        assert!(check_command("nvidia-smi -pl 150").is_err());
        assert!(check_command("systemctl status lemond").is_ok());
        assert!(check_command("systemctl restart lemond").is_err());
        assert!(check_command("rm -rf /tmp/x").is_err());
        assert!(check_command("/bin/cat /etc/hostname").is_err());
        assert!(check_command("find /tmp -delete").is_err());
        assert!(check_command("git log --oneline -3").is_ok());
        assert!(check_command("git branch newbranch").is_err());
        assert!(check_command("git push").is_err());
        assert!(check_command("curl -s http://127.0.0.1:5560/health").is_ok());
        assert!(check_command("curl -s https://example.com").is_err());
        assert!(check_command("curl -d x http://127.0.0.1:5560/board/message").is_err());
        assert!(check_command("curl -sXPOST http://127.0.0.1:5560/board/message").is_err());
        assert!(check_command("curl -s example.com").is_err());
        assert!(check_command("curl -s -m 5 -H 'Accept: x' 10.1.2.3:5560/health").is_ok());
        assert!(check_command("journalctl --vacuum-time=1s").is_err());
        assert!(check_command("hostname evil").is_err());
        assert!(check_command("ip route add default via 1.2.3.4").is_err());
        assert!(check_command("cat /home/someone/.ssh/id_ed25519").is_err());
        assert!(check_command("cat ~/.bashrc").is_err());
        assert!(check_command("python3 -c 1").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn secrets_refused_after_resolving() {
        let dir = std::env::temp_dir().join(format!("ori-policy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let secret = dir.join("api_token.txt");
        std::fs::write(&secret, "x").unwrap();
        let link = dir.join("innocent");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&secret, &link).unwrap();
        assert!(check_read_path(link.to_str().unwrap()).is_err(), "a symlink to a secret is still a secret");
        let plain = dir.join("notes.txt");
        std::fs::write(&plain, "x").unwrap();
        assert!(check_read_path(plain.to_str().unwrap()).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_size_rule() {
        assert_eq!(params_b("tiel-coder-35b-a3b"), Some(35.0));
        assert_eq!(params_b("Qwen3-VL-4B-Instruct-GGUF"), Some(4.0));
        assert_eq!(params_b("Qwen3-0.6B-GGUF"), Some(0.6));
        assert_eq!(params_b("bitnet-b1.58-2B-4T"), Some(2.0));
        assert_eq!(params_b("Hermes-3-8B"), Some(8.0));
        assert_eq!(params_b("katcoder"), None);
        assert!(action_allowed("minimaxai/minimax-m3.1-flash", &[]), "NIM");
        assert!(action_allowed("Qwen3-VL-4B-Instruct-GGUF", &[]));
        assert!(!action_allowed("Qwen3-0.6B-GGUF", &[]));
        assert!(!action_allowed("bitnet-b1.58-2B-4T", &[]), "2B is not over 2B");
        assert!(!action_allowed("katcoder", &[]));
        assert!(action_allowed("katcoder", &["katcoder".to_string()]));
    }

    #[test]
    fn action_guard() {
        assert!(check_action_command("cp a.conf a.conf.bak && sed -i 's/x/y/' a.conf && systemctl --user restart foo").is_ok());
        assert!(check_action_command("git log --format=%su -3").is_ok());
        assert!(check_action_command("sudo systemctl restart lemond").is_err());
        assert!(check_action_command("echo x | /usr/bin/sudo -S true").is_err());
        assert!(check_action_command("ssh node2 'sudo reboot'").is_err());
        assert!(check_action_command("ssh node2 'systemctl --user status foo'").is_ok());
        assert!(check_action_command("ssh 127.0.0.1 id").is_err(), "back into this machine");
        assert!(check_action_command("ssh localhost id").is_err(), "back into this machine");
        assert!(check_action_command("docker run -v /:/h alpine").is_err());
        assert!(check_action_command("cat ~/.config/ori/tokens.json").is_err());
        assert!(check_action_command("cat /home/someone/.ssh/id_ed25519").is_err());
    }

    #[test]
    fn action_shell_cannot_sudo() {
        if Command::new("setpriv").arg("--version").output().is_err() {
            return; // not on this machine
        }
        let (out, _) = run_action("id -u; sudo -n true 2>&1 | head -1", "/tmp", Duration::from_secs(10)).unwrap();
        assert!(!out.contains("\n0\n") && !out.starts_with("0\n"), "{out}");
        assert!(out.contains("no new privileges") || out.contains("password") || out.contains("not allowed"), "{out}");
        let (out, code) = run_action("sleep 30 & echo started", "/tmp", Duration::from_secs(10)).unwrap();
        assert_eq!(code, 0);
        assert!(out.contains("started") || out.contains("background process"), "{out}");
    }

    #[test]
    fn pipeline_runs_and_times_out() {
        let s = check_command("uname -s | tr a-z A-Z").unwrap();
        let (out, code) = run_pipeline(&s, Duration::from_secs(5)).unwrap();
        assert_eq!(code, 0);
        assert_eq!(out.trim(), "LINUX");
        let slow = vec![vec!["tail".to_string(), "-f".to_string(), "/dev/null".to_string()]];
        assert!(run_pipeline(&slow, Duration::from_millis(300)).is_err());
    }
}
