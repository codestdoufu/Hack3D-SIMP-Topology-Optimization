import { useState, useRef, useEffect } from "react";
import "./App.css";

const API = "http://127.0.0.1:5000";

const DEFAULT_PARAMS = {
  nx: 20, ny: 6, nz: 4,
  volumeFraction: 0.2,
  penalty: 3.0,
  iterations: 30,
  fixedFace: "x0",
  loadFace: "x1",
  loadDirection: -1,
  loadMagnitude: 10000,
  threshold: 0.5,
};

// ── PRESETS ──────────────────────────────────────────────────────────────────
const PRESETS = [
  {
    label: "Cantilever",
    icon: "⊣",
    desc: "Fixed left, load right",
    params: { nx: 20, ny: 6, nz: 4, volumeFraction: 0.2, penalty: 3.0, iterations: 60, fixedFace: "x0", loadFace: "x1", loadDirection: -1, loadMagnitude: 10000, threshold: 0.5 },
  },
  {
    label: "Bridge",
    icon: "⌒",
    desc: "Fixed bottom, load top-center",
    params: { nx: 24, ny: 8, nz: 4, volumeFraction: 0.3, penalty: 3.0, iterations: 80, fixedFace: "y0", loadFace: "y1", loadDirection: -1, loadMagnitude: 20000, threshold: 0.5 },
  },
  {
    label: "Column",
    icon: "⬆",
    desc: "Fixed base, vertical load",
    params: { nx: 6, ny: 20, nz: 6, volumeFraction: 0.25, penalty: 3.5, iterations: 60, fixedFace: "y0", loadFace: "y1", loadDirection: -1, loadMagnitude: 50000, threshold: 0.5 },
  },
  {
    label: "Quick Test",
    icon: "⚡",
    desc: "Fast low-res preview",
    params: { nx: 10, ny: 4, nz: 3, volumeFraction: 0.2, penalty: 3.0, iterations: 30, fixedFace: "x0", loadFace: "x1", loadDirection: -1, loadMagnitude: 10000, threshold: 0.5 },
  },
];

// ── TOOLTIPS ─────────────────────────────────────────────────────────────────
const TIPS = {
  "Elements X": "Number of voxel elements along X. More = finer detail but slower.",
  "Elements Y": "Number of voxel elements along Y axis.",
  "Elements Z": "Number of voxel elements along Z axis.",
  "Volume Fraction": "What fraction of the design space gets filled with material. 0.2 = 20% material.",
  "SIMP Penalty": "Penalises intermediate densities, pushing material to be fully solid or void. Higher = crisper result.",
  "Iterations": "How many optimization steps to run. More = converges better, takes longer.",
  "Density Threshold": "Elements below this density are hidden in the 3D view.",
  "Load Magnitude": "Total applied force in Newtons.",
  "Fixed Face": "The face of the structure that is rigidly clamped — it cannot move.",
  "Load Face": "The face where the external force is applied.",
  "Load Direction": "Which direction the force pushes.",
};

const FACES = [
  { value: "x0", label: "X₀" }, { value: "x1", label: "X₁" },
  { value: "y0", label: "Y₀" }, { value: "y1", label: "Y₁" },
  { value: "z0", label: "Z₀" }, { value: "z1", label: "Z₁" },
];

const ATTACKS = [
  { value: "noise",    label: "Gaussian Noise",   desc: "Adds random perturbation" },
  { value: "scale",    label: "Density Scaling",  desc: "Multiplies all values by 0.9" },
  { value: "zero",     label: "Random Zeroing",   desc: "Zeros 20% of elements" },
  { value: "quantize", label: "Quantization",     desc: "Reduces to 5 density levels" },
  { value: "smooth",   label: "Smoothing",        desc: "Moving-average blur" },
];

function estimateTime(p) {
  const s = Math.round(p.nx * p.ny * p.nz * p.iterations * 0.008);
  return s < 60 ? `~${s}s` : `~${Math.floor(s / 60)}m ${s % 60}s`;
}

// ── TOOLTIP ──────────────────────────────────────────────────────────────────
function Tooltip({ text }) {
  const [show, setShow] = useState(false);
  return (
    <span className="tip-wrap" onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      <span className="tip-icon">?</span>
      {show && <span className="tip-box">{text}</span>}
    </span>
  );
}

function SliderField({ label, name, min, max, step, value, onChange, unit = "" }) {
  return (
    <div className="field">
      <div className="field-header">
        <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span className="field-label">{label}</span>
          {TIPS[label] && <Tooltip text={TIPS[label]} />}
        </span>
        <span className="field-value">{value}{unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(name, step < 1 ? parseFloat(e.target.value) : parseInt(e.target.value))}
        className="slider" />
      <div className="slider-bounds"><span>{min}</span><span>{max}</span></div>
    </div>
  );
}

function SelectField({ label, name, options, value, onChange }) {
  return (
    <div className="field">
      <span style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 5 }}>
        <span className="field-label" style={{ marginBottom: 0 }}>{label}</span>
        {TIPS[label] && <Tooltip text={TIPS[label]} />}
      </span>
      <div className="select-row">
        {options.map(opt => (
          <button key={opt.value}
            className={`face-btn ${value === opt.value ? "active" : ""}`}
            onClick={() => onChange(name, opt.value)}>{opt.label}</button>
        ))}
      </div>
    </div>
  );
}

// ── PRESET BAR ───────────────────────────────────────────────────────────────
function PresetBar({ onSelect }) {
  return (
    <div className="preset-bar">
      {PRESETS.map(p => (
        <button key={p.label} className="preset-btn" onClick={() => onSelect(p.params)} title={p.desc}>
          <span className="preset-icon">{p.icon}</span>
          <span className="preset-label">{p.label}</span>
        </button>
      ))}
    </div>
  );
}

// ── BC DIAGRAM ────────────────────────────────────────────────────────────────
function BcDiagram({ fixedFace, loadFace, loadDirection }) {
  // Maps face value to which SVG face to highlight
  const faceColors = { x0: "#1a2a3a", x1: "#1a2a3a", y0: "#1a2a3a", y1: "#1a2a3a", z0: "#1a2a3a", z1: "#1a2a3a" };
  const fixedColor = "rgba(0,229,255,0.25)";
  const loadColor  = "rgba(255,107,53,0.3)";
  const fc = { ...faceColors };
  if (fixedFace) fc[fixedFace] = fixedColor;
  if (loadFace)  fc[loadFace]  = loadColor;

  // Arrow direction for load
  const arrowDir = loadDirection === -1 ? "↓" : "↑";

  return (
    <div className="bc-diagram">
      <svg viewBox="0 0 160 100" xmlns="http://www.w3.org/2000/svg" style={{ width: "100%", height: "auto" }}>
        {/* Isometric box */}
        {/* Bottom face */}
        <polygon points="30,70 80,85 130,70 80,55" fill={fc["y0"]} stroke="#2a3540" strokeWidth="1"/>
        {/* Top face */}
        <polygon points="30,30 80,45 130,30 80,15" fill={fc["y1"]} stroke="#2a3540" strokeWidth="1"/>
        {/* Left face (x0) */}
        <polygon points="30,30 30,70 80,85 80,45" fill={fc["x0"]} stroke="#2a3540" strokeWidth="1"/>
        {/* Right face (x1) */}
        <polygon points="130,30 130,70 80,85 80,45" fill={fc["x1"]} stroke="#2a3540" strokeWidth="1"/>
        {/* Front face (z0) - small left sliver */}
        <polygon points="30,30 30,70 32,69 32,31" fill={fc["z0"]} stroke="#2a3540" strokeWidth="0.5"/>
        {/* Back face (z1) - small right sliver */}
        <polygon points="130,30 130,70 128,69 128,31" fill={fc["z1"]} stroke="#2a3540" strokeWidth="0.5"/>

        {/* Fixed face hatch lines */}
        {fixedFace === "x0" && [0,1,2,3,4].map(i => (
          <line key={i} x1={28} y1={33 + i*9} x2={18} y2={38 + i*9} stroke="var(--accent)" strokeWidth="1" opacity="0.6"/>
        ))}
        {fixedFace === "x1" && [0,1,2,3,4].map(i => (
          <line key={i} x1={132} y1={33 + i*9} x2={142} y2={38 + i*9} stroke="var(--accent)" strokeWidth="1" opacity="0.6"/>
        ))}
        {fixedFace === "y0" && [0,1,2,3,4].map(i => (
          <line key={i} x1={35 + i*12} y1={88} x2={30 + i*12} y2={96} stroke="var(--accent)" strokeWidth="1" opacity="0.6"/>
        ))}

        {/* Load arrow */}
        {loadFace === "x1" && (
          <text x="148" y="52" fill="var(--accent2)" fontSize="14" textAnchor="middle">{arrowDir}</text>
        )}
        {loadFace === "x0" && (
          <text x="12" y="52" fill="var(--accent2)" fontSize="14" textAnchor="middle">{arrowDir}</text>
        )}
        {loadFace === "y1" && (
          <text x="80" y="10" fill="var(--accent2)" fontSize="14" textAnchor="middle">{arrowDir}</text>
        )}
        {loadFace === "y0" && (
          <text x="80" y="98" fill="var(--accent2)" fontSize="14" textAnchor="middle">{arrowDir}</text>
        )}

        {/* Labels */}
        <text x="22" y="52" fill="var(--accent)" fontSize="7" opacity="0.7" fontFamily="monospace">X₀</text>
        <text x="133" y="52" fill="var(--accent)" fontSize="7" opacity="0.7" fontFamily="monospace">X₁</text>
        <text x="76" y="94" fill="var(--accent)" fontSize="7" opacity="0.7" fontFamily="monospace">Y₀</text>
        <text x="76" y="13" fill="var(--accent)" fontSize="7" opacity="0.7" fontFamily="monospace">Y₁</text>
      </svg>
      <div className="bc-legend">
        <span className="bc-legend-item"><span className="bc-swatch" style={{ background: "rgba(0,229,255,0.4)" }} />Fixed</span>
        <span className="bc-legend-item"><span className="bc-swatch" style={{ background: "rgba(255,107,53,0.5)" }} />Load</span>
      </div>
    </div>
  );
}

// ── EXPORT BUTTON ─────────────────────────────────────────────────────────────
function ExportBtn({ result, params }) {
  const [open, setOpen] = useState(false);

  const downloadImg = (b64, filename) => {
    const a = document.createElement("a");
    a.href = "data:image/png;base64," + b64;
    a.download = filename;
    a.click();
  };

  const exportAll = () => {
    if (!result) return;
    downloadImg(result.images.structure["0.5"], "structure_rho05.png");
    setTimeout(() => downloadImg(result.images.convergence, "convergence.png"), 300);
    setTimeout(() => downloadImg(result.images.histogram,   "density_histogram.png"), 600);
    setOpen(false);
  };

  if (!result) return null;
  return (
    <div className="export-wrap">
      <button className="export-btn" onClick={() => setOpen(o => !o)}>⬇ EXPORT</button>
      {open && (
        <div className="export-menu">
          <button onClick={() => { downloadImg(result.images.structure["0.1"], "structure_rho01.png"); setOpen(false); }}>Structure ρ&gt;0.1</button>
          <button onClick={() => { downloadImg(result.images.structure["0.3"], "structure_rho03.png"); setOpen(false); }}>Structure ρ&gt;0.3</button>
          <button onClick={() => { downloadImg(result.images.structure["0.5"], "structure_rho05.png"); setOpen(false); }}>Structure ρ&gt;0.5</button>
          <button onClick={() => { downloadImg(result.images.convergence, "convergence.png"); setOpen(false); }}>Convergence plot</button>
          <button onClick={() => { downloadImg(result.images.histogram,   "density_histogram.png"); setOpen(false); }}>Density histogram</button>
          <div className="export-divider"/>
          <button onClick={exportAll}>⬇ Download all (3 files)</button>
        </div>
      )}
    </div>
  );
}

function AnimatedNumber({ value, decimals = 0, suffix = "" }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    if (value == null) return;
    const end = parseFloat(value);
    const start = Date.now();
    const tick = () => {
      const t = Math.min((Date.now() - start) / 900, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplay(end * eased);
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [value]);
  return <>{display.toFixed(decimals)}{suffix}</>;
}

function IterRow({ iter }) {
  return (
    <div className="iter-row">
      <span className="iter-n">{String(iter.iteration).padStart(3, "0")}</span>
      <span className="iter-c">{iter.compliance.toExponential(3)}</span>
      <span className="iter-v">{(iter.volume * 100).toFixed(2)}%</span>
      <span className="iter-d">{iter.density_change.toExponential(2)}</span>
    </div>
  );
}

function Toast({ toasts }) {
  return (
    <div className="toast-container">
      {toasts.map(t => (
        <div key={t.id} className={`toast toast-${t.kind}`}>
          <span className="toast-icon">{t.kind === "ok" ? "✓" : t.kind === "err" ? "✗" : "◈"}</span>
          {t.msg}
        </div>
      ))}
    </div>
  );
}

function ScoreBar({ score }) {
  const pct   = Math.min(100, Math.max(0, score));
  const color = pct > 70 ? "var(--green)" : pct > 40 ? "var(--orange)" : "#ff4444";
  return (
    <div className="score-bar-wrap">
      <div className="score-bar-track">
        <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="score-bar-label" style={{ color }}>{pct.toFixed(1)}%</span>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("optimizer");
  const [params, setParams]       = useState(DEFAULT_PARAMS);

  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [elapsed, setElapsed]     = useState(null);
  const [statusMsg, setStatusMsg] = useState("");
  const [iterLog, setIterLog]     = useState([]);
  const [progress, setProgress]   = useState(0);
  const [activeThreshold, setActiveThreshold] = useState("0.5");
  const logRef = useRef(null);
  const t0Ref  = useRef(null);

  const [origDensity, setOrigDensity]       = useState(null);
  const [wmDensity, setWmDensity]           = useState(null);
  const [wmMessage, setWmMessage]           = useState("NYU-HACK3D");
  const [wmAlpha, setWmAlpha]               = useState(0.03);
  const [wmKey, setWmKey]                   = useState("hack3d-nyu-vip-2025");
  const [wmEmbedResult, setWmEmbedResult]   = useState(null);
  const [wmDetectResult, setWmDetectResult] = useState(null);
  const [wmAttackResult, setWmAttackResult] = useState(null);
  const [wmAttack, setWmAttack]             = useState("noise");
  const [wmLoading, setWmLoading]           = useState(false);
  const [wmStep, setWmStep]                 = useState("idle");

  const [toasts, setToasts] = useState([]);
  const toastId = useRef(0);
  const addToast = (msg, kind = "info") => {
    const id = ++toastId.current;
    setToasts(t => [...t, { id, msg, kind }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 3500);
  };

  const handleChange = (name, value) => setParams(p => ({ ...p, [name]: value }));
  const handlePreset = (presetParams) => { setParams(presetParams); addToast(`Preset loaded`, "info"); };

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [iterLog]);

  const handleRun = () => {
    setLoading(true); setError(null); setResult(null);
    setIterLog([]); setProgress(0); setStatusMsg("Connecting…");
    t0Ref.current = Date.now();

    fetch(`${API}/optimize/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }).then(res => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      const pump = () => reader.read().then(({ done, value }) => {
        if (done) { setLoading(false); return; }
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const evt = JSON.parse(line.slice(6));
            if (evt.type === "status") {
              setStatusMsg(evt.msg);
            } else if (evt.type === "iteration") {
              setIterLog(prev => [...prev.slice(-199), evt]);
              setProgress(evt.pct);
              setStatusMsg(`Iter ${evt.iteration + 1}/${evt.total} · C: ${evt.compliance.toExponential(3)}`);
            } else if (evt.type === "done") {
              setResult(evt);
              setElapsed(((Date.now() - t0Ref.current) / 1000).toFixed(1));
              setProgress(100); setStatusMsg("Complete ✓"); setLoading(false);
              setOrigDensity(evt.density); setWmDensity(evt.density);
              setWmStep("idle"); setWmEmbedResult(null); setWmDetectResult(null); setWmAttackResult(null);
              addToast("Optimization complete — density field ready", "ok");
            } else if (evt.type === "error") {
              setError(evt.msg); setLoading(false);
              addToast("Backend error: " + evt.msg, "err");
            }
          } catch (_) {}
        }
        pump();
      }).catch(e => { setError(e.message); setLoading(false); });
      pump();
    }).catch(() => {
      setError(`Cannot reach Flask at ${API} — run: python app.py`);
      setLoading(false);
    });
  };

  const handleEmbed = async () => {
    if (!origDensity) return;
    setWmLoading(true);
    try {
      const res = await fetch(`${API}/watermark/embed`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ density: origDensity, message: wmMessage, alpha: wmAlpha, secretKey: wmKey }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error);
      setWmEmbedResult(data); setWmDensity(data.watermarked_density);
      setWmStep("embedded"); setWmDetectResult(null); setWmAttackResult(null);
      addToast(`Watermark embedded — SNR ${data.snr_db} dB`, "ok");
    } catch (e) { addToast("Embed error: " + e.message, "err"); }
    finally { setWmLoading(false); }
  };

  const handleDetect = async () => {
    setWmLoading(true);
    try {
      const res = await fetch(`${API}/watermark/detect`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ density: wmDensity, original_density: origDensity, secretKey: wmKey, n_bits: wmEmbedResult?.n_bits || 64 }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error);
      setWmDetectResult(data); setWmStep("detected");
      addToast(data.is_watermarked ? "Watermark verified ✓" : "Watermark not detected ✗", data.is_watermarked ? "ok" : "err");
    } catch (e) { addToast("Detect error: " + e.message, "err"); }
    finally { setWmLoading(false); }
  };

  const handleAttack = async () => {
    if (!wmDensity) return;
    setWmLoading(true);
    try {
      const res = await fetch(`${API}/watermark/attack`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ density: wmDensity, original_density: origDensity, attack: wmAttack, secretKey: wmKey }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error);
      setWmAttackResult(data); setWmStep("attacked");
      addToast(data.is_watermarked_after_attack ? "Watermark survived the attack ✓" : "Attack destroyed watermark ✗", data.is_watermarked_after_attack ? "ok" : "err");
    } catch (e) { addToast("Attack error: " + e.message, "err"); }
    finally { setWmLoading(false); }
  };

  const PIPE_STEPS  = ["idle", "embedded", "detected", "attacked"];
  const PIPE_LABELS = { idle: "READY", embedded: "EMBEDDED", detected: "VERIFIED", attacked: "ATTACKED" };
  const currentStepIdx = PIPE_STEPS.indexOf(wmStep);
  const elemCount = params.nx * params.ny * params.nz;

  return (
    <div className="app">
      <div className="scanlines" />
      <Toast toasts={toasts} />

      <header className="header">
        <div className="header-left">
          <div className="logo-mark">H3</div>
          <div>
            <h1 className="title">HACK3D</h1>
            <p className="subtitle">SIMP Topology Optimizer · Digital Manufacturing Security</p>
          </div>
        </div>
        <div className="header-center">
          <button className={`nav-tab ${activeTab === "optimizer" ? "nav-active" : ""}`} onClick={() => setActiveTab("optimizer")}>▶ OPTIMIZER</button>
          <button className={`nav-tab ${activeTab === "watermark" ? "nav-active" : ""}`} onClick={() => setActiveTab("watermark")}>
            ◈ WATERMARK LAB {origDensity && <span className="nav-badge">●</span>}
          </button>
        </div>
        <div className="header-right">
          <span className="tag">NYU VIP</span>
          <span className="tag">FEM · SIMP</span>
        </div>
      </header>

      <main className="main">
        <aside className="panel controls-panel">

          {activeTab === "optimizer" && <>
            <div className="panel-title"><span className="dot" /> PARAMETERS</div>
            <div className="section-label">QUICK PRESETS</div>
            <PresetBar onSelect={handlePreset} />
            <div className="divider" />
            <div className="section-label">MESH RESOLUTION</div>
            <SliderField label="Elements X" name="nx" min={5}  max={40} step={1} value={params.nx} onChange={handleChange} />
            <SliderField label="Elements Y" name="ny" min={2}  max={12} step={1} value={params.ny} onChange={handleChange} />
            <SliderField label="Elements Z" name="nz" min={2}  max={8}  step={1} value={params.nz} onChange={handleChange} />
            <div className="info-row">
              <span>{elemCount.toLocaleString()} elements</span>
              <span style={{ color: "var(--accent2)" }}>{estimateTime(params)}</span>
            </div>
            <div className="divider" />
            <div className="section-label">OPTIMIZER</div>
            <SliderField label="Volume Fraction" name="volumeFraction" min={0.1} max={0.8} step={0.01} value={params.volumeFraction} onChange={handleChange} />
            <SliderField label="SIMP Penalty"    name="penalty"        min={1.0} max={5.0} step={0.1}  value={params.penalty}        onChange={handleChange} />
            <SliderField label="Iterations"      name="iterations"     min={10}  max={200} step={10}   value={params.iterations}     onChange={handleChange} />
            <div className="divider" />
            <div className="section-label">BOUNDARY CONDITIONS</div>
            <BcDiagram fixedFace={params.fixedFace} loadFace={params.loadFace} loadDirection={params.loadDirection} />
            <SelectField label="Fixed Face" name="fixedFace" options={FACES} value={params.fixedFace} onChange={handleChange} />
            <SelectField label="Load Face"  name="loadFace"  options={FACES} value={params.loadFace}  onChange={handleChange} />
            <div className="field">
              <div className="field-header">
                <span className="field-label">Load Direction</span>
                <span className="field-value">{params.loadDirection === -1 ? "−Y (down)" : "+Y (up)"}</span>
              </div>
              <div className="select-row">
                {[{ value: -1, label: "↓ Down" }, { value: 1, label: "↑ Up" }].map(o => (
                  <button key={o.value} className={`face-btn ${params.loadDirection === o.value ? "active" : ""}`}
                    onClick={() => handleChange("loadDirection", o.value)}>{o.label}</button>
                ))}
              </div>
            </div>
            <SliderField label="Load Magnitude" name="loadMagnitude" min={1000} max={100000} step={1000} value={params.loadMagnitude} onChange={handleChange} unit=" N" />
            <div className="divider" />
            <div className="section-label">DISPLAY</div>
            <SliderField label="Density Threshold" name="threshold" min={0.1} max={0.9} step={0.1} value={params.threshold} onChange={handleChange} />
            <button className="run-btn" onClick={handleRun} disabled={loading}>
              {loading ? <><span className="spinner" /> COMPUTING…</> : <><span className="run-icon">▶</span> RUN OPTIMIZATION</>}
            </button>
            {loading && <>
              <div className="info-row" style={{ padding: "4px 14px 2px" }}>
                <span style={{ fontFamily: "var(--mono)", fontSize: "10px", color: "var(--text-dim)", maxWidth: "70%", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{statusMsg}</span>
                <span style={{ fontFamily: "var(--mono)", fontSize: "10px", color: "var(--accent)" }}>{progress.toFixed(0)}%</span>
              </div>
              <div className="progress-bar-wrap">
                <div className="progress-bar-fill" style={{ width: `${progress}%`, transition: "width 0.3s ease" }} />
              </div>
            </>}
          </>}

          {activeTab === "watermark" && <>
            <div className="panel-title">
              <span className="dot" style={{ background: "var(--accent2)", boxShadow: "0 0 6px var(--accent2)" }} />
              WATERMARK CONFIG
            </div>
            {!origDensity && (
              <div className="wm-no-data"><span>⚠</span><span>Run the optimizer first to generate a density field.</span></div>
            )}
            <div className="section-label">SIGNATURE</div>
            <div className="field">
              <span className="field-label">Message to Embed</span>
              <input className="wm-input" value={wmMessage} onChange={e => setWmMessage(e.target.value)} placeholder="NYU-HACK3D" disabled={!origDensity} />
            </div>
            <div className="field">
              <span className="field-label">Secret Key</span>
              <input className="wm-input" value={wmKey} onChange={e => setWmKey(e.target.value)} placeholder="secret key" type="password" disabled={!origDensity} />
            </div>
            <div className="field">
              <div className="field-header">
                <span className="field-label">Strength (α)</span>
                <span className="field-value">{wmAlpha.toFixed(3)}</span>
              </div>
              <input type="range" min={0.005} max={0.1} step={0.005} value={wmAlpha}
                onChange={e => setWmAlpha(parseFloat(e.target.value))} className="slider" disabled={!origDensity} />
              <div className="slider-bounds"><span>subtle</span><span>robust</span></div>
            </div>
            <div className="divider" />
            <div className="section-label">PIPELINE</div>
            <button className="wm-step-btn" onClick={handleEmbed} disabled={!origDensity || wmLoading}>
              <span className="wm-step-n">01</span>
              <span className="wm-step-text">
                <span>EMBED WATERMARK</span>
                <span className="wm-step-desc">Spread-spectrum into density field</span>
              </span>
              {wmStep !== "idle" && <span className="wm-check">✓</span>}
            </button>
            <button className="wm-step-btn" onClick={handleDetect} disabled={wmStep === "idle" || wmLoading}>
              <span className="wm-step-n">02</span>
              <span className="wm-step-text">
                <span>DETECT WATERMARK</span>
                <span className="wm-step-desc">Carrier correlation + decode</span>
              </span>
              {(wmStep === "detected" || wmStep === "attacked") && <span className="wm-check">✓</span>}
            </button>
            <div className="divider" />
            <div className="section-label">ATTACK SIMULATION</div>
            <div className="field">
              <span className="field-label">Attack Type</span>
              <div className="attack-list">
                {ATTACKS.map(a => (
                  <button key={a.value} className={`attack-btn ${wmAttack === a.value ? "active" : ""}`} onClick={() => setWmAttack(a.value)}>
                    <span className="attack-label">{a.label}</span>
                    <span className="attack-desc">{a.desc}</span>
                  </button>
                ))}
              </div>
            </div>
            <button className="wm-step-btn accent2" onClick={handleAttack} disabled={wmStep === "idle" || wmLoading}>
              <span className="wm-step-n" style={{ color: "var(--accent2)" }}>03</span>
              <span className="wm-step-text">
                <span>SIMULATE ATTACK</span>
                <span className="wm-step-desc">Test watermark robustness</span>
              </span>
              {wmStep === "attacked" && <span className="wm-check" style={{ color: "var(--accent2)" }}>✓</span>}
            </button>
            {wmLoading && (
              <div className="wm-loading-row">
                <span className="spinner" style={{ borderTopColor: "var(--accent)" }} /> Processing…
              </div>
            )}
          </>}
        </aside>

        <section className="results-area">

          {activeTab === "optimizer" && <>
            {!result && !loading && !error && (
              <div className="empty-state">
                <div className="empty-grid" />
                <div className="empty-icon">⬡</div>
                <p className="empty-title">Configure parameters and run the optimizer</p>
                <p className="empty-sub">
                  Live iteration feed · 3D structure · Convergence plots<br />
                  <span className="empty-tip">TIP: Start at 30 iterations (~45s) to verify connectivity</span>
                </p>
              </div>
            )}
            {error && (
              <div className="error-card">
                <span className="error-icon">⚠</span>
                <div>
                  <strong>Backend Error</strong>
                  <p>{error}</p>
                  <p className="error-hint">1. Run <code>python app.py</code> &nbsp; 2. Check <code>http://127.0.0.1:5000/health</code></p>
                </div>
              </div>
            )}
            {(loading || iterLog.length > 0) && (
              <div className="panel result-panel">
                <div className="panel-title" style={{ justifyContent: "space-between" }}>
                  <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span className="dot" style={{ animation: loading ? "pulse 1s infinite" : "none" }} />
                    LIVE ITERATION FEED
                  </span>
                  {loading && <span className="live-pct">{progress.toFixed(0)}%</span>}
                </div>
                <div className="iter-header">
                  <span>ITER</span><span>COMPLIANCE</span><span>VOLUME</span><span>Δρ MAX</span>
                </div>
                <div className="iter-log" ref={logRef}>
                  {iterLog.map(it => <IterRow key={it.iteration} iter={it} />)}
                  {loading && <div className="iter-cursor">▋</div>}
                </div>
              </div>
            )}
            {result && <>
              <div className="metrics-bar">
                <div className="metric"><span className="metric-label">COMPLIANCE</span><span className="metric-value" style={{ fontSize: 14 }}>{result.metrics.finalCompliance.toExponential(3)}</span></div>
                <div className="metric"><span className="metric-label">VOLUME</span><span className="metric-value"><AnimatedNumber value={result.metrics.finalVolume * 100} decimals={1} suffix="%" /></span></div>
                <div className="metric"><span className="metric-label">ITERATIONS</span><span className="metric-value"><AnimatedNumber value={result.metrics.iterations} decimals={0} /></span></div>
                <div className="metric" style={{ position: "relative" }}>
                  <span className="metric-label">COMPUTE TIME</span>
                  <span className="metric-value">{elapsed}s</span>
                  <ExportBtn result={result} params={params} />
                </div>
              </div>
              <div className="panel result-panel">
                <div className="panel-title"><span className="dot green" /> 3D OPTIMIZED STRUCTURE</div>
                <div className="threshold-tabs">
                  {["0.1", "0.3", "0.5"].map(t => (
                    <button key={t} className={`tab-btn ${activeThreshold === t ? "active" : ""}`} onClick={() => setActiveThreshold(t)}>ρ &gt; {t}</button>
                  ))}
                </div>
                {result.images.structure[activeThreshold] && (
                  <img className="result-img" src={`data:image/png;base64,${result.images.structure[activeThreshold]}`} alt="structure" />
                )}
              </div>
              <div className="panel result-panel">
                <div className="panel-title"><span className="dot blue" /> CONVERGENCE HISTORY</div>
                <img className="result-img" src={`data:image/png;base64,${result.images.convergence}`} alt="convergence" />
              </div>
              <div className="panel result-panel">
                <div className="panel-title"><span className="dot orange" /> DENSITY DISTRIBUTION</div>
                <img className="result-img" src={`data:image/png;base64,${result.images.histogram}`} alt="histogram" />
              </div>
              <button className="wm-cta" onClick={() => setActiveTab("watermark")}>
                <span>◈</span><span>Density field ready — open Watermark Lab →</span>
              </button>
            </>}
          </>}

          {activeTab === "watermark" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {!origDensity ? (
                <div className="empty-state">
                  <div className="empty-grid" />
                  <div className="empty-icon">◈</div>
                  <p className="empty-title">No density field available</p>
                  <p className="empty-sub">Run the optimizer first to generate a topology, then embed and test watermarks here.</p>
                  <button className="run-btn" style={{ width: 260, marginTop: 16 }} onClick={() => setActiveTab("optimizer")}>← GO TO OPTIMIZER</button>
                </div>
              ) : <>
                <div className="wm-pipeline">
                  {PIPE_STEPS.map((s, i) => (
                    <div key={s} className={`wm-pipe-step ${i < currentStepIdx ? "done" : ""} ${i === currentStepIdx ? "active" : ""}`}>
                      <span className="pipe-n">{i < currentStepIdx ? "✓" : i + 1}</span>
                      <span className="pipe-label">{PIPE_LABELS[s]}</span>
                    </div>
                  ))}
                </div>
                {wmEmbedResult && (
                  <div className="panel result-panel">
                    <div className="panel-title">
                      <span className="dot" style={{ background: "var(--accent2)", boxShadow: "0 0 6px var(--accent2)" }} />
                      WATERMARK EMBEDDED
                    </div>
                    <div className="wm-stat-grid">
                      <div className="wm-stat"><span className="wm-stat-label">MESSAGE</span><span className="wm-stat-val">"{wmEmbedResult.message}"</span></div>
                      <div className="wm-stat"><span className="wm-stat-label">BITS</span><span className="wm-stat-val">{wmEmbedResult.n_bits}</span></div>
                      <div className="wm-stat"><span className="wm-stat-label">STRENGTH α</span><span className="wm-stat-val">{wmEmbedResult.alpha}</span></div>
                      <div className="wm-stat"><span className="wm-stat-label">SNR</span><span className="wm-stat-val">{wmEmbedResult.snr_db} dB</span></div>
                    </div>
                    <img className="result-img" src={`data:image/png;base64,${wmEmbedResult.image}`} alt="embed" />
                  </div>
                )}
                {wmDetectResult && (
                  <div className="panel result-panel">
                    <div className="panel-title">
                      <span className="dot" style={{ background: wmDetectResult.is_watermarked ? "var(--green)" : "#ff4444", boxShadow: `0 0 6px ${wmDetectResult.is_watermarked ? "var(--green)" : "#ff4444"}` }} />
                      DETECTION — {wmDetectResult.is_watermarked ? "✓ WATERMARK VERIFIED" : "✗ NOT DETECTED"}
                    </div>
                    <div className="wm-stat-grid">
                      <div className="wm-stat"><span className="wm-stat-label">DECODED MESSAGE</span><span className="wm-stat-val">"{wmDetectResult.detected_message}"</span></div>
                      <div className="wm-stat"><span className="wm-stat-label">CONFIDENCE</span><span className="wm-stat-val">{(wmDetectResult.avg_confidence * 100).toFixed(1)}%</span></div>
                    </div>
                    <div style={{ padding: "10px 16px 14px" }}>
                      <div style={{ fontFamily: "var(--mono)", fontSize: "9px", color: "var(--text-dim)", letterSpacing: "0.15em", marginBottom: 6 }}>CORRELATION SCORE</div>
                      <ScoreBar score={wmDetectResult.correlation_score} />
                    </div>
                    <img className="result-img" src={`data:image/png;base64,${wmDetectResult.image}`} alt="detect" />
                  </div>
                )}
                {wmAttackResult && (
                  <div className="panel result-panel">
                    <div className="panel-title">
                      <span className="dot" style={{ background: "#ff4444", boxShadow: "0 0 6px #ff4444" }} />
                      ATTACK — {wmAttackResult.attack_meta.attack.toUpperCase()}
                    </div>
                    <div className="wm-stat-grid">
                      <div className="wm-stat"><span className="wm-stat-label">DISTORTION RMS</span><span className="wm-stat-val">{wmAttackResult.attack_meta.distortion_rms}</span></div>
                      <div className="wm-stat">
                        <span className="wm-stat-label">STATUS</span>
                        <span className="wm-stat-val" style={{ color: wmAttackResult.is_watermarked_after_attack ? "var(--green)" : "#ff4444" }}>
                          {wmAttackResult.is_watermarked_after_attack ? "✓ SURVIVED" : "✗ DESTROYED"}
                        </span>
                      </div>
                      <div className="wm-stat"><span className="wm-stat-label">DECODED</span><span className="wm-stat-val">"{wmAttackResult.detected_message}"</span></div>
                    </div>
                    <div style={{ padding: "10px 16px 14px" }}>
                      <div style={{ fontFamily: "var(--mono)", fontSize: "9px", color: "var(--text-dim)", letterSpacing: "0.15em", marginBottom: 6 }}>POST-ATTACK SCORE</div>
                      <ScoreBar score={wmAttackResult.correlation_score} />
                    </div>
                    <img className="result-img" src={`data:image/png;base64,${wmAttackResult.image}`} alt="attack" />
                  </div>
                )}
              </>}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}