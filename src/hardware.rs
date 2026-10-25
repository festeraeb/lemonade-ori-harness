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
    pub vram_mb: Option<u64>,
}

pub fn detect_hardware() -> HardwareProfile {
    let host_name = std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "localhost".into());

    let os = std::env::consts::OS.to_string();
    let cpu_cores = num_cpus();
    let total_memory_gb = estimate_total_ram_gb();

    let mut gpus = Vec::new();

    // 1. Check for NVIDIA CUDA via nvidia-smi
    if let Ok(output) = Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            for line in text.lines() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if !parts.is_empty() && !parts[0].is_empty() {
                    let vram = parts.get(1).and_then(|v| v.parse::<u64>().ok());
                    gpus.push(GpuDevice {
                        name: parts[0].to_string(),
                        backend: "cuda".into(),
                        vram_mb: vram,
                    });
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
