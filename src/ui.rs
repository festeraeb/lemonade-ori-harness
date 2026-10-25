pub fn render_html(host_name: &str, backend: &str) -> String {
    format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lemonade Ori — Standalone Agent Terminal</title>
  <style>
    :root {{
      --bg: #0b0f14;
      --card: #141b24;
      --border: #222e3d;
      --accent: #f59e0b;
      --accent-glow: rgba(245, 158, 11, 0.25);
      --text: #e2e8f0;
      --muted: #64748b;
      --green: #10b981;
      --blue: #38bdf8;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      background: var(--card);
      border-bottom: 1px solid var(--border);
      padding: 0.75rem 1.25rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}
    .lemon-badge {{
      background: var(--accent);
      color: #000;
      font-weight: 800;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-size: 0.85rem;
    }}
    .title {{ font-size: 1.1rem; font-weight: 700; letter-spacing: 0.05em; }}
    .status-pill {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.8rem;
      color: var(--muted);
      background: rgba(0,0,0,0.3);
      padding: 0.3rem 0.6rem;
      border-radius: 999px;
      border: 1px solid var(--border);
    }}
    .dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--green); }}
    .modes {{
      display: flex;
      gap: 0.5rem;
      padding: 0.5rem 1.25rem;
      background: rgba(20, 27, 36, 0.6);
      border-bottom: 1px solid var(--border);
    }}
    .mode-btn {{
      background: transparent;
      border: 1px solid var(--border);
      color: var(--muted);
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.8rem;
      font-weight: 600;
      transition: all 0.15s ease;
    }}
    .mode-btn.active {{
      background: var(--accent-glow);
      border-color: var(--accent);
      color: var(--accent);
    }}
    main {{
      flex: 1;
      overflow-y: auto;
      padding: 1rem 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }}
    .msg {{
      max-width: 85%;
      padding: 0.75rem 1rem;
      border-radius: 8px;
      font-size: 0.9rem;
      line-height: 1.5;
      white-space: pre-wrap;
    }}
    .msg.user {{
      align-self: flex-end;
      background: #1e293b;
      border: 1px solid #334155;
    }}
    .msg.assistant {{
      align-self: flex-start;
      background: var(--card);
      border: 1px solid var(--border);
    }}
    .msg.meta {{
      align-self: center;
      background: transparent;
      color: var(--muted);
      font-size: 0.8rem;
      border: 1px dashed var(--border);
    }}
    footer {{
      background: var(--card);
      border-top: 1px solid var(--border);
      padding: 0.75rem 1.25rem;
      display: flex;
      gap: 0.75rem;
    }}
    textarea {{
      flex: 1;
      background: #0f172a;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.6rem 0.8rem;
      border-radius: 6px;
      font-family: inherit;
      font-size: 0.9rem;
      resize: none;
      height: 48px;
    }}
    textarea:focus {{ outline: 1px solid var(--accent); }}
    button.send-btn {{
      background: var(--accent);
      color: #000;
      border: none;
      font-weight: 700;
      padding: 0 1.5rem;
      border-radius: 6px;
      cursor: pointer;
    }}
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span class="lemon-badge">LEMONADE</span>
      <span class="title">ORI HARNESS</span>
    </div>
    <div class="status-pill">
      <span class="dot"></span>
      <span>{host_name} ({backend})</span>
    </div>
  </header>

  <div class="modes">
    <button class="mode-btn active" data-mode="chat">Chat</button>
    <button class="mode-btn" data-mode="plan">Plan</button>
    <button class="mode-btn" data-mode="vibe">Vibe</button>
    <button class="mode-btn" data-mode="eval">Eval</button>
    <button class="mode-btn" data-mode="bench">Challenge Benchmark</button>
  </div>

  <main id="log">
    <div class="msg meta">Lemonade Ori initialized. Standalone mode ready. MCP surface active at /mcp.</div>
  </main>

  <footer>
    <textarea id="prompt" placeholder="Ask Lemonade Ori (Shift+Enter for newline, Enter to submit)..."></textarea>
    <button class="send-btn" id="send">Send</button>
  </footer>

  <script>
    let activeMode = 'chat';
    const log = document.getElementById('log');
    const prompt = document.getElementById('prompt');
    const send = document.getElementById('send');
    const modeBtns = document.querySelectorAll('.mode-btn');

    modeBtns.forEach(btn => {{
      btn.addEventListener('click', () => {{
        modeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeMode = btn.dataset.mode;
        if (activeMode === 'bench') {{
          runBenchmark();
        }}
      }});
    }});

    function addMsg(role, text) {{
      const d = document.createElement('div');
      d.className = `msg ${{role}}`;
      d.textContent = text;
      log.appendChild(d);
      log.scrollTop = log.scrollHeight;
      return d;
    }}

    async function runBenchmark() {{
      addMsg('meta', 'Running Lemonade Challenge evaluation suite...');
      const res = await fetch('/api/bench', {{ method: 'POST' }});
      const data = await res.json();
      addMsg('assistant', '🏆 LEMONADE CHALLENGE SCORECARD:\n' + JSON.stringify(data, null, 2));
    }}

    async function sendPrompt() {{
      const text = prompt.value.trim();
      if (!text) return;
      prompt.value = '';
      addMsg('user', text);

      const bot = addMsg('assistant', 'Thinking...');
      try {{
        const res = await fetch('/v1/chat/completions', {{
          method: 'POST',
          headers: {{
            'Content-Type': 'application/json',
            'X-Lemonade-Mode': activeMode
          }},
          body: JSON.stringify({{
            messages: [{{ role: 'user', content: text }}],
            stream: false
          }})
        }});
        const data = await res.json();
        const content = data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
        bot.textContent = content || JSON.stringify(data);
      }} catch (err) {{
        bot.textContent = 'Error: ' + err.message;
      }}
    }}

    send.addEventListener('click', sendPrompt);
    prompt.addEventListener('keydown', (e) => {{
      if (e.key === 'Enter' && !e.shiftKey) {{
        e.preventDefault();
        sendPrompt();
      }}
    }});
  </script>
</body>
</html>
"##, host_name = host_name, backend = backend)
}
