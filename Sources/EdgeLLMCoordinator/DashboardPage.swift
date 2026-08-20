/// The dashboard served at "/". Static HTML plus a small script that polls
/// /api/status once per second. No external assets, works on any browser on
/// the local network, including phones.
enum DashboardPage {
    static let html = #"""
    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>edge-llm-cluster</title>
    <style>
      :root {
        --bg: #0e1116; --card: #171c24; --line: #262d38;
        --text: #e6e9ee; --dim: #8b95a5; --accent: #4da3ff;
        --ok: #3fca6b; --warn: #e0b13d; --bad: #e05d5d;
      }
      * { box-sizing: border-box; margin: 0; }
      body {
        background: var(--bg); color: var(--text);
        font: 15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif;
        padding: 24px; max-width: 1080px; margin: 0 auto;
      }
      header { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
      h1 { font-size: 20px; font-weight: 650; }
      .model { color: var(--dim); font-size: 14px; }
      .stats { display: flex; gap: 12px; margin: 18px 0 22px; flex-wrap: wrap; }
      .stat {
        background: var(--card); border: 1px solid var(--line);
        border-radius: 10px; padding: 10px 16px; min-width: 130px;
      }
      .stat b { display: block; font-size: 22px; font-weight: 650; }
      .stat span { color: var(--dim); font-size: 12.5px; }
      .grid {
        display: grid; gap: 14px;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      }
      .card {
        background: var(--card); border: 1px solid var(--line);
        border-radius: 12px; padding: 16px;
      }
      .card h2 { font-size: 16px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
      .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--ok); }
      .dot.offline { background: var(--bad); }
      .meta { color: var(--dim); font-size: 13px; margin: 4px 0 12px; }
      .row { display: flex; justify-content: space-between; font-size: 13.5px; margin-top: 8px; }
      .row .k { color: var(--dim); }
      .bar {
        height: 8px; border-radius: 4px; background: var(--line);
        overflow: hidden; margin-top: 6px;
      }
      .bar i { display: block; height: 100%; background: var(--accent); border-radius: 4px; }
      .bar i.warn { background: var(--warn); }
      .bar i.bad { background: var(--bad); }
      .empty {
        color: var(--dim); border: 1px dashed var(--line); border-radius: 12px;
        padding: 40px; text-align: center;
      }
      footer { color: var(--dim); font-size: 12.5px; margin-top: 26px; }
    </style>
    </head>
    <body>
      <header>
        <h1>edge-llm-cluster</h1>
        <span class="model" id="model"></span>
      </header>
      <div class="stats">
        <div class="stat"><b id="deviceCount">0</b><span>devices</span></div>
        <div class="stat"><b id="tps">0.0</b><span>tokens / second</span></div>
        <div class="stat"><b id="tokens">0</b><span>tokens generated</span></div>
        <div class="stat"><b id="layers">0</b><span>model layers</span></div>
      </div>
      <div class="grid" id="grid"></div>
      <div class="empty" id="empty">Waiting for workers to join the cluster.</div>
      <footer>Auto refreshes every second. Serve address: this host, port shown in the coordinator log.</footer>
    <script>
      const fmtGB = b => (b / 1073741824).toFixed(1) + " GB";
      const platformLabel = p => ({
        macos: "macOS", ios: "iPhone", ipados: "iPad",
        android: "Android", linux: "Linux"
      }[p] || p);

      async function refresh() {
        let s;
        try {
          s = await (await fetch("/api/status")).json();
        } catch { return; }

        document.getElementById("model").textContent = s.model || "no model loaded";
        document.getElementById("deviceCount").textContent = s.devices.length;
        document.getElementById("tps").textContent = (s.tokensPerSecond || 0).toFixed(1);
        document.getElementById("tokens").textContent = s.generatedTokens || 0;
        document.getElementById("layers").textContent = s.totalLayers || 0;
        document.getElementById("empty").style.display = s.devices.length ? "none" : "block";

        const now = Date.now() / 1000;
        document.getElementById("grid").innerHTML = s.devices.map(d => {
          const offline = now - d.lastSeen > 10;
          const used = d.usedMemory || 0;
          const quota = d.info.usableMemory || 1;
          const pct = Math.min(100, used / quota * 100);
          const cls = pct > 90 ? "bad" : pct > 75 ? "warn" : "";
          const range = d.layerStart != null
            ? `layers ${d.layerStart} to ${d.layerEnd - 1}` : "no shard assigned";
          return `
            <div class="card">
              <h2><span class="dot ${offline ? "offline" : ""}"></span>${d.info.name}</h2>
              <div class="meta">${platformLabel(d.info.platform)} &middot; ${d.info.runtime} &middot; ${range}</div>
              <div class="row"><span class="k">memory quota</span><span>${fmtGB(used)} / ${fmtGB(quota)}</span></div>
              <div class="bar"><i class="${cls}" style="width:${pct}%"></i></div>
              <div class="row"><span class="k">total memory</span><span>${fmtGB(d.info.totalMemory)}</span></div>
              <div class="row"><span class="k">per token</span><span>${(d.millisecondsPerToken || 0).toFixed(1)} ms</span></div>
              <div class="row"><span class="k">status</span><span>${offline ? "offline" : "online"}</span></div>
            </div>`;
        }).join("");
      }

      refresh();
      setInterval(refresh, 1000);
    </script>
    </body>
    </html>
    """#
}
