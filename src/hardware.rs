use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub host_name: String,
    pub os: String,
    pub cpu_cores: usize,
    pub total_memory_gb: f64,
    pub gpus: Vec<GpuDevice>,
    pub recommended_backend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub name: String,
    pub backend: String,
    /// Total VRAM in MB where available (NVIDIA total, macOS shared, Vulkan unknown).
    pub vram_mb: Option<u64>,
    /// Currently free VRAM in MB where measurable (NVIDIA); None where unknown (Vulkan/CPU).
    pub vram_free_mb: Option<u64>,
}

impl GpuDevice {
    /// Free VRAM in MB; `None` when not measurable (Vulkan/CPU).
    pub fn free_mb(&self) -> Option<u64> {
        self.vram_free_mb
    }
}

pub fn detect_hardware() -> HardwareProfile {
    let host_name = std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "localhost".into());

    let os = std::env::consts::OS.to_string();
    let cpu_cores = num_cpus();
    let total_memory_gb = estimate_total_ram_gb();

    let mut gpus = Vec::new();

    // 1. Check for NVIDIA CUDA via nvidia-smi (total + used -> free VRAM)
    if let Ok(output) = Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total,memory.used")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            for line in text.lines() {
                if let Some(g) = parse_nvidia_line(line) {
                    gpus.push(g);
                }
            }
        }
    }

    // 2. Check for Vulkan on Linux DRI / AMD / Intel
    if gpus.is_empty() && std::path::Path::new("/dev/dri").exists() {
        if let Ok(entries) = std::fs::read_dir("/dev/dri") {
            let renders: Vec<_> = entries
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with("renderD"))
                .collect();
            if !renders.is_empty() {
                gpus.push(GpuDevice {
                    name: format!("Generic Vulkan Device ({} render nodes)", renders.len()),
                    backend: "vulkan".into(),
                    vram_free_mb: None,
                    vram_mb: None,
                });
            }
        }
    }

    // 3. Check for Apple Metal on macOS
    if gpus.is_empty() && os == "macos" {
        gpus.push(GpuDevice {
            name: "Apple Silicon Metal GPU".into(),
            backend: "metal".into(),
            vram_free_mb: None,
            vram_mb: Some((total_memory_gb * 1024.0) as u64),
        });
    }

    let recommended_backend = if let Some(gpu) = gpus.first() {
        gpu.backend.clone()
    } else {
        "cpu".into()
    };

    HardwareProfile {
        host_name,
        os,
        cpu_cores,
        total_memory_gb,
        gpus,
        recommended_backend,
    }
}

/// Parse one `nvidia-smi --format=csv,noheader,nounits` line: `name, total, used`.
/// Returns `None` for blank/unparseable lines; free VRAM = total.saturating_sub(used).
pub fn parse_nvidia_line(line: &str) -> Option<GpuDevice> {
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
    if parts.is_empty() || parts[0].is_empty() {
        return None;
    }
    let total = parts.get(1).and_then(|v| v.parse::<u64>().ok());
    let used = parts.get(2).and_then(|v| v.parse::<u64>().ok());
    let free = total.and_then(|t| used.map(|u| t.saturating_sub(u)));
    Some(GpuDevice {
        name: parts[0].to_string(),
        backend: "cuda".into(),
        vram_mb: total,
        vram_free_mb: free,
    })
}

impl HardwareProfile {
    /// Total free VRAM across GPUs where it is measurable, in MB (0 if unknown).
    pub fn free_vram_mb(&self) -> u64 {
        self.gpus.iter().filter_map(|g| g.vram_free_mb).sum()
    }

    /// The GPU with the most currently-free VRAM (local seat selection for "open/free vram").
    pub fn freest_gpu(&self) -> Option<&GpuDevice> {
        self.gpus
            .iter()
            .filter(|g| g.vram_free_mb.is_some())
            .max_by_key(|g| g.vram_free_mb.unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nvidia_smi_free() {
        let g = parse_nvidia_line("Tesla P100-PCIE-16GB, 16384, 4200").unwrap();
        assert_eq!(g.name, "Tesla P100-PCIE-16GB");
        assert_eq!(g.backend, "cuda");
        assert_eq!(g.vram_mb, Some(16384));
        assert_eq!(g.vram_free_mb, Some(12184));
        assert_eq!(g.free_mb(), Some(12184));

        // used > total is clamped to 0 free
        let g2 = parse_nvidia_line("T4, 16384, 99999").unwrap();
        assert_eq!(g2.vram_free_mb, Some(0));

        // blank line -> None
        assert!(parse_nvidia_line("").is_none());
        assert!(parse_nvidia_line("   ").is_none());

        let prof = HardwareProfile {
            host_name: "h".into(),
            os: "linux".into(),
            cpu_cores: 4,
            total_memory_gb: 16.0,
            gpus: vec![g, g2],
            recommended_backend: "cuda".into(),
        };
        assert_eq!(prof.free_vram_mb(), 12184);
        assert_eq!(prof.freest_gpu().map(|d| d.name.as_str()), Some("Tesla P100-PCIE-16GB"));
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn estimate_total_ram_gb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if let Some(kb) = parts.get(1).and_then(|k| k.parse::<f64>().ok()) {
                        return (kb / (1024.0 * 1024.0) * 10.0).round() / 10.0;
                    }
                }
            }
        }
    }
    16.0 // Sensible default fallback
}
