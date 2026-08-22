/* SAF v4 — read-only terminal client.
   PRINCIPLE: the browser is an EYE, never a brain.
   - All data comes from the FastAPI server (no local PRNG, no fake signals).
   - Every free-text value rendered via textContent (XSS-safe — PART 10).
   - Every panel shows a provenance badge from the server's own metadata.
*/
const API = "http://127.0.0.1:8000";

function el(id) { return document.getElementById(id); }

// Safe text setter — NEVER innerHTML on server/feed content.
function setText(id, txt) { el(id).textContent = txt; }

function provBadge(source) {
  const map = { live: ["b-live", "LIVE"], cached: ["b-cached", "CACHED"],
                est: ["b-sim", "EST"], sim: ["b-sim", "SIM"] };
  const [cls, label] = map[source] || ["b-sim", (source || "SIM").toUpperCase()];
  const s = document.createElement("span");
  s.className = "badge " + cls;
  s.textContent = label;                 // textContent, not innerHTML
  return s;
}

function pctCell(v) {
  const td = document.createElement("td");
  if (v === null || v === undefined) { td.textContent = "—"; return td; }
  td.textContent = (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
  td.className = v >= 0 ? "pos" : "neg";
  return td;
}

async function loadHealth() {
  try {
    const r = await fetch(API + "/api/system/health");
    const d = await r.json();
    el("health-badge").appendChild(provBadge(d.provenance?.source));
    const rows = [
      ["Status", d.status],
      ["AI key (server-side)", d.ai_key_present ? "present ✅" : "MISSING ⚠️"],
      ["Benchmark", `${d.benchmark} (${d.benchmark_bars} bars, last ${d.benchmark_last})`],
      ["Baskets", d.baskets],
      ["Universe tickers", d.universe_tickers],
      ["Audit chain", d.audit_chain_ok ? "intact ✅" : "BROKEN ❌"],
    ];
    const box = el("health"); box.innerHTML = "";
    for (const [k, v] of rows) {
      const div = document.createElement("div"); div.className = "kv";
      const a = document.createElement("span"); a.textContent = k;
      const b = document.createElement("span"); b.textContent = String(v);
      div.appendChild(a); div.appendChild(b); box.appendChild(div);
    }
    setText("prov", "provenance: " + JSON.stringify(d.provenance));
    setText("status", "");
  } catch (e) {
    setText("status", "❌ Cannot reach server. Is `python -m saf.server` running?");
  }
}

async function loadScreen() {
  try {
    const r = await fetch(API + "/api/screen?top=15");
    const d = await r.json();
    el("screen-badge").appendChild(provBadge(d.provenance?.source));
    const body = el("screen-body"); body.innerHTML = "";
    for (const row of d.top) {
      const c = row.components || {};
      const tr = document.createElement("tr");
      const tck = document.createElement("td"); tck.textContent = row.ticker;
      const sc  = document.createElement("td"); sc.textContent = row.total; sc.className = "pos";
      const t = document.createElement("td"); t.textContent = c.trend ?? "—";
      const a = document.createElement("td"); a.textContent = c.alpha_indep ?? "—";
      const rr = document.createElement("td"); rr.textContent = c.rel_strength ?? "—";
      const vd = document.createElement("td");
      const vb = document.createElement("span"); vb.className = "verdict v-" + row.verdict;
      vb.textContent = row.verdict; vd.appendChild(vb);
      const cf = document.createElement("td"); cf.textContent = row.confidence || "—";
      tr.append(tck, sc, t, a, rr, vd, cf);
      body.appendChild(tr);
    }
  } catch (e) {
    el("screen-body").innerHTML =
      '<tr><td colspan="7" style="color:#ef4444">screen failed — check server</td></tr>';
  }
}

loadHealth();
loadScreen();
// Refresh every 5 min. Read-only: no writes, no keys, no simulation.
setInterval(() => { loadHealth(); loadScreen(); }, 300000);
