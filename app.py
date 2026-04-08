"""
app.py — Flask backend for the Hack3D SIMP Topology Optimizer.

Exposes three groups of endpoints:
  1. /optimize/stream  — SSE stream that runs SIMP and pushes per-iteration
                         progress to the UI in real time, then returns images.
  2. /watermark/*      — Embed, detect, and attack-test digital watermarks in
                         FEM density fields (cybersecurity research module).
  3. /health           — Simple liveness check.

Run with:
    python app.py
Then open http://localhost:3000 in the React frontend.
"""

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — renders to memory, no GUI window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from matplotlib import cm
import base64
import io
import json
import time
import threading

from fem3d_numpy import HexFEMSolver3D
from simp_numpy import SIMPOptimizer
from watermark import DensityWatermark

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})   # Allow all origins for local dev


# ── SHARED STATE ──────────────────────────────────────────────────────────────
# Stores the current optimization run's live progress so the SSE stream can
# read from it. Single-user dev server only — not safe for concurrent users.
_run_state = {
    "running":  False,
    "progress": [],     # One dict per completed iteration
    "result":   None,   # Final result dict set when optimization finishes
    "error":    None,   # Error message string if something goes wrong
}
_state_lock = threading.Lock()   # Guards reads/writes to _run_state


# ── HELPERS ───────────────────────────────────────────────────────────────────

def fig_to_base64(fig):
    """
    Serialize a matplotlib Figure to a base64-encoded PNG string.
    React displays it directly: <img src={`data:image/png;base64,${b64}`} />
    Closes the figure after encoding to free memory.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64


def plot_3d_design(nodes, elems, density, threshold=0.5, title="3D Structure"):
    """
    Render the optimized 3D structure as a hexahedral voxel plot.

    Only elements with density > threshold are drawn, colored on a
    Red-Yellow-Green scale: red = borderline density, green = solid.

    Args:
        nodes     : (N_nodes, 3) array of node coordinates
        elems     : (N_elems, 8) array of element node indices (hex connectivity)
        density   : (N_elems,)   array of element densities in [0, 1]
        threshold : elements below this density are hidden
        title     : plot title string
    """
    fig = plt.figure(figsize=(10, 7), facecolor='#0d1117')
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor('#0d1117')

    active_elems = np.where(density > threshold)[0]

    if len(active_elems) == 0:
        ax.text(0.5, 0.5, 0.5, f"No elements with density > {threshold}",
                transform=ax.transAxes, ha="center", fontsize=12, color='#c8d8e8')
        ax.set_title(title, color='#c8d8e8')
        return fig

    cmap = matplotlib.colormaps["RdYlGn"]
    norm = Normalize(vmin=threshold, vmax=1.0)

    # Draw each active element's 6 faces as a colored polygon collection
    for elem_idx in active_elems:
        elem_nodes  = elems[elem_idx]
        elem_coords = nodes[elem_nodes]
        color = cmap(norm(density[elem_idx]))

        # Hexahedral face index sets (bottom, top, front, back, left, right)
        faces = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5]]
        for face in faces:
            ax.add_collection3d(Poly3DCollection(
                [elem_coords[face]], facecolors=color, edgecolors='k', linewidths=0.3
            ))

    # Fit axis limits and preserve true geometric proportions
    all_coords = nodes[elems[active_elems]].reshape(-1, 3)
    ax.set_xlim([all_coords[:,0].min(), all_coords[:,0].max()])
    ax.set_ylim([all_coords[:,1].min(), all_coords[:,1].max()])
    ax.set_zlim([all_coords[:,2].min(), all_coords[:,2].max()])
    ax.set_box_aspect([np.ptp(all_coords[:,0]), np.ptp(all_coords[:,1]), np.ptp(all_coords[:,2])])

    ax.set_xlabel("X (m)", color='#5a7080')
    ax.set_ylabel("Y (m)", color='#5a7080')
    ax.set_zlabel("Z (m)", color='#5a7080')
    ax.tick_params(colors='#5a7080')
    ax.set_title(title, color='#c8d8e8', pad=10)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("Material Density", color='#8a9db0')
    cbar.ax.yaxis.set_tick_params(color='#5a7080')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#5a7080')
    plt.tight_layout()
    return fig


def build_fem(data):
    """
    Construct and configure a HexFEMSolver3D from a JSON request payload.

    Applies boundary conditions (fixed face) and a distributed load based on
    the face labels sent from the React UI (e.g. 'x0', 'y1', 'z0').

    Returns:
        fem : configured HexFEMSolver3D instance ready for SIMPOptimizer
    """
    nx         = int(data.get('nx', 20))
    ny         = int(data.get('ny', 6))
    nz         = int(data.get('nz', 4))
    fixed_face = data.get('fixedFace', 'x0')
    load_face  = data.get('loadFace', 'x1')
    load_dir   = int(data.get('loadDirection', -1))
    load_mag   = float(data.get('loadMagnitude', 1e4))

    fem = HexFEMSolver3D(E_mod=200e9, nu=0.3)   # Steel: E=200 GPa, Poisson's ratio=0.3
    fem.set_mesh(Lx=1.0, Ly=0.2, Lz=0.1, nx=nx, ny=ny, nz=nz)

    # Map face string labels → (axis_index, coordinate) pairs
    face_map = {
        'x0': (0, 0.0), 'x1': (0, 1.0),
        'y0': (1, 0.0), 'y1': (1, 0.2),
        'z0': (2, 0.0), 'z1': (2, 0.1),
    }

    bc_axis, bc_coord     = face_map.get(fixed_face, (0, 0.0))
    fem.fix_face(axis=bc_axis, coord=bc_coord)

    load_axis, load_coord = face_map.get(load_face, (0, 1.0))
    fem.add_distributed_load(
        axis=load_axis, coord=load_coord,
        direction=load_dir,
        total=load_mag / (fem.ny * fem.nz)   # Distribute total force over all face nodes
    )
    return fem


def build_plots(fem, density, volume_fraction, history):
    """
    Generate all result plots and return as base64 PNG strings.

    Returns:
        conv_img       : convergence history (compliance, volume fraction, Δρ)
        structure_imgs : dict of { threshold_str: base64_png } for ρ > 0.1/0.3/0.5
        hist_img       : density distribution histogram
    """
    # ── Convergence history (3 subplots) ──────────────────────────────────────
    fig_conv, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='#0d1117')
    for ax in axes:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#5a7080')
        ax.spines[:].set_color('#1e2730')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color('#5a7080')

    iters = history['iteration']
    axes[0].semilogy(iters, history['compliance'], color='#4d9cff', lw=2, marker='o', ms=3)
    axes[0].set_title("Compliance Convergence", color='#c8d8e8')
    axes[0].set_xlabel("Iteration", color='#5a7080')
    axes[0].grid(True, alpha=0.15, color='#1e2730')

    axes[1].plot(iters, history['volume'], color='#39ff8a', lw=2, marker='o', ms=3, label='Actual')
    axes[1].axhline(volume_fraction, color='#ff6b35', linestyle='--', lw=2, label=f'Target ({volume_fraction:.2f})')
    axes[1].set_title("Volume Constraint", color='#c8d8e8')
    axes[1].set_xlabel("Iteration", color='#5a7080')
    axes[1].legend(facecolor='#0f1318', edgecolor='#1e2730', labelcolor='#c8d8e8')
    axes[1].grid(True, alpha=0.15, color='#1e2730')

    axes[2].semilogy(iters, history['density_change'], color='#ff9f43', lw=2, marker='o', ms=3)
    axes[2].set_title("Convergence Indicator", color='#c8d8e8')
    axes[2].set_xlabel("Iteration", color='#5a7080')
    axes[2].grid(True, alpha=0.15, color='#1e2730')

    plt.tight_layout()
    conv_img = fig_to_base64(fig_conv)

    # ── 3D structure at multiple density thresholds ────────────────────────────
    structure_imgs = {}
    for t in [0.1, 0.3, 0.5]:
        fig3d = plot_3d_design(fem.nodes_np, fem.elems_t, density,
                               threshold=t, title=f"Optimized design (density > {t})")
        structure_imgs[str(t)] = fig_to_base64(fig3d)

    # ── Density histogram ──────────────────────────────────────────────────────
    # A good SIMP result shows a bimodal distribution:
    # most elements near 0 (void) or 1 (solid), few in the "gray" zone.
    fig_hist, ax = plt.subplots(figsize=(8, 5), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#5a7080')
    ax.spines[:].set_color('#1e2730')
    ax.hist(density, bins=30, color='#4d9cff', edgecolor='#0d1117', alpha=0.8)
    ax.axvline(np.mean(density), color='#ff6b35', linestyle='--', lw=2, label=f'Mean: {np.mean(density):.3f}')
    ax.axvline(volume_fraction,  color='#39ff8a', linestyle='--', lw=2, label=f'Target: {volume_fraction:.3f}')
    ax.set_xlabel("Density", color='#5a7080')
    ax.set_ylabel("Elements", color='#5a7080')
    ax.set_title("Density Distribution", color='#c8d8e8')
    ax.legend(facecolor='#0f1318', edgecolor='#1e2730', labelcolor='#c8d8e8')
    ax.grid(True, alpha=0.15, color='#1e2730')
    plt.tight_layout()
    hist_img = fig_to_base64(fig_hist)

    return conv_img, structure_imgs, hist_img


# ── OPTIMIZE — SSE STREAM ─────────────────────────────────────────────────────

@app.route('/optimize/stream', methods=['POST'])
def optimize_stream():
    """
    Run SIMP topology optimization and stream results as Server-Sent Events.

    Each SSE line starts with "data: " followed by a JSON object:
      { type: 'status',    msg }
      { type: 'iteration', iteration, compliance, volume, density_change, pct, total }
      { type: 'done',      metrics, images, density }
      { type: 'error',     msg }

    The React frontend reads the stream and updates the live feed,
    progress bar, and result panels in real time.
    """
    data        = request.json
    volume_frac = float(data.get('volumeFraction', 0.2))
    penalty     = float(data.get('penalty', 3.0))
    iterations  = int(data.get('iterations', 30))

    def generate():
        try:
            # Step 1 — Build FEM mesh and apply boundary/load conditions
            yield f"data: {json.dumps({'type':'status','msg':'Building FEM mesh…'})}\n\n"
            fem = build_fem(data)

            # Step 2 — Initialize SIMP optimizer
            yield f"data: {json.dumps({'type':'status','msg':'Initializing SIMP optimizer…'})}\n\n"
            optimizer = SIMPOptimizer(
                fem_solver=fem,
                initial_density=volume_frac,
                volume_fraction=volume_frac,
                penalty=penalty,
                filter_radius=0.02,   # Density filter prevents checkerboarding artifacts
            )

            history = {'iteration': [], 'compliance': [], 'volume': [], 'density_change': []}

            # Step 3 — Run iterations, streaming each result to the UI
            yield f"data: {json.dumps({'type':'status','msg':f'Running {iterations} iterations…'})}\n\n"
            for i in range(iterations):
                result = optimizer.step()   # Single SIMP iteration

                history['iteration'].append(i)
                history['compliance'].append(float(result['compliance']))
                history['volume'].append(float(result['volume']))
                history['density_change'].append(float(result['density_change']))

                yield f"data: {json.dumps({'type':'iteration','iteration':i,'compliance':result['compliance'],'volume':result['volume'],'density_change':result['density_change'],'pct':round((i+1)/iterations*100,1),'total':iterations})}\n\n"

                # Early stop if density field has converged
                if result['density_change'] < 1e-3 and i > 20:
                    yield f"data: {json.dumps({'type':'status','msg':f'Converged at iteration {i}'})}\n\n"
                    break

            # Step 4 — Render result images
            yield f"data: {json.dumps({'type':'status','msg':'Rendering result images…'})}\n\n"
            density = optimizer.get_density()
            conv_img, structure_imgs, hist_img = build_plots(fem, density, volume_frac, history)

            # Step 5 — Send final done event with all data the UI needs
            yield f"data: {json.dumps({'type':'done','metrics':{'finalCompliance':round(float(history['compliance'][-1]),6),'finalVolume':round(float(history['volume'][-1]),4),'iterations':len(history['iteration'])},'images':{'convergence':conv_img,'structure':structure_imgs,'histogram':hist_img},'density':density.tolist()})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type':'error','msg':str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control':     'no-cache',
            'X-Accel-Buffering': 'no',   # Disable Nginx proxy buffering for true streaming
        }
    )


# ── WATERMARK ENDPOINTS ───────────────────────────────────────────────────────

@app.route('/watermark/embed', methods=['POST'])
def watermark_embed():
    """
    Embed a spread-spectrum watermark into a FEM density field.

    Expects JSON: { density, message, alpha, secretKey }
    Returns:      { watermarked_density, snr_db, alpha, message, n_bits, image }
    """
    try:
        data       = request.json
        density    = np.array(data['density'], dtype=float)
        message    = data.get('message', 'NYU-HACK3D')
        alpha      = float(data.get('alpha', 0.03))
        secret_key = data.get('secretKey', 'hack3d-nyu-vip-2025')

        wm     = DensityWatermark(secret_key=secret_key, alpha=alpha)
        result = wm.embed(density, message=message)

        perturbation = np.array(result['perturbation'])
        n = len(density)
        x = np.arange(n)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='#0d1117')
        for ax in axes:
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#5a7080')
            ax.spines[:].set_color('#1e2730')

        # Left: original vs watermarked density overlay
        axes[0].bar(x, density,                      color='#4d9cff', alpha=0.6, width=1.0, label='Original')
        axes[0].bar(x, result['watermarked_density'], color='#39ff8a', alpha=0.4, width=1.0, label='Watermarked')
        axes[0].set_title("Density Comparison", color='#c8d8e8')
        axes[0].set_xlabel("Element Index", color='#5a7080')
        axes[0].legend(facecolor='#0f1318', edgecolor='#1e2730', labelcolor='#c8d8e8')
        axes[0].grid(True, alpha=0.1, color='#1e2730')

        # Right: the watermark perturbation signal (green = +, orange = -)
        axes[1].plot(x, perturbation, color='#ff6b35', lw=0.8, alpha=0.9)
        axes[1].axhline(0, color='#1e2730', lw=1)
        axes[1].fill_between(x, perturbation, 0, where=(perturbation > 0), color='#39ff8a', alpha=0.3)
        axes[1].fill_between(x, perturbation, 0, where=(perturbation < 0), color='#ff6b35', alpha=0.3)
        axes[1].set_title(f"Watermark Signal (α={alpha})", color='#c8d8e8')
        axes[1].set_xlabel("Element Index", color='#5a7080')
        axes[1].set_ylabel("Perturbation", color='#5a7080')
        axes[1].grid(True, alpha=0.1, color='#1e2730')

        plt.tight_layout()

        return jsonify({
            'success':             True,
            'watermarked_density': result['watermarked_density'].tolist(),
            'snr_db':              result['snr_db'],
            'alpha':               alpha,
            'message':             message,
            'n_bits':              result['n_bits'],
            'image':               fig_to_base64(fig),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/watermark/detect', methods=['POST'])
def watermark_detect():
    """
    Detect and decode a watermark from a density field.

    Uses informed detection — correlates the recovered perturbation against
    the known carrier sequence generated from the secret key.

    Expects JSON: { density, original_density, secretKey, n_bits }
    Returns:      { is_watermarked, detected_message, correlation_score,
                    avg_confidence, image }
    """
    try:
        data             = request.json
        density          = np.array(data['density'], dtype=float)
        original_density = np.array(data.get('original_density', data['density']), dtype=float)
        secret_key       = data.get('secretKey', 'hack3d-nyu-vip-2025')
        n_bits           = int(data.get('n_bits', 64))

        wm     = DensityWatermark(secret_key=secret_key)
        result = wm.detect(density, original=original_density, n_bits=n_bits)

        # Per-bit confidence bar chart — green = detected above threshold, orange = missed
        conf   = result['confidence']
        colors = ['#39ff8a' if c > 0.1 else '#ff6b35' for c in conf]

        fig, ax = plt.subplots(figsize=(12, 3), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#5a7080')
        ax.spines[:].set_color('#1e2730')
        ax.bar(range(len(conf)), conf, color=colors, width=0.8)
        ax.axhline(0.1, color='#00e5ff', linestyle='--', lw=1.5, label='Detection threshold')
        ax.set_title(
            f"Bit Confidence | Score: {result['correlation_score']}% | "
            f"{'✓ WATERMARK DETECTED' if result['is_watermarked'] else '✗ NOT DETECTED'}",
            color='#39ff8a' if result['is_watermarked'] else '#ff6b35'
        )
        ax.set_xlabel("Bit Index", color='#5a7080')
        ax.set_ylabel("Correlation", color='#5a7080')
        ax.legend(facecolor='#0f1318', edgecolor='#1e2730', labelcolor='#c8d8e8')
        ax.grid(True, alpha=0.1, color='#1e2730')
        plt.tight_layout()

        return jsonify({
            'success':           True,
            'is_watermarked':    result['is_watermarked'],
            'detected_message':  result['detected_message'],
            'correlation_score': result['correlation_score'],
            'avg_confidence':    result['avg_confidence'],
            'image':             fig_to_base64(fig),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/watermark/attack', methods=['POST'])
def watermark_attack():
    """
    Simulate an adversarial attack on a watermarked density field, then
    re-detect to measure how much of the watermark survived.

    Attack types: noise, scale, zero, quantize, smooth
    See watermark.py → DensityWatermark.simulate_attack for full details.

    Expects JSON: { density, original_density, attack, secretKey, ...params }
    Returns:      { attack_meta, is_watermarked_after_attack,
                    correlation_score, detected_message, image }
    """
    try:
        data             = request.json
        density          = np.array(data['density'], dtype=float)
        original_density = np.array(data['original_density'], dtype=float)
        attack           = data.get('attack', 'noise')
        secret_key       = data.get('secretKey', 'hack3d-nyu-vip-2025')

        attack_params = {
            'sigma':    float(data.get('sigma',    0.05)),
            'factor':   float(data.get('factor',   0.9)),
            'fraction': float(data.get('fraction', 0.2)),
            'n_levels': int(data.get('n_levels',   5)),
            'window':   int(data.get('window',     5)),
        }

        wm = DensityWatermark(secret_key=secret_key)

        # Apply attack, then detect what remains
        attack_result = wm.simulate_attack(density, attack=attack, **attack_params)
        attacked      = attack_result['attacked_density']
        detect_result = wm.detect(attacked, original=original_density)

        # Three-panel plot: original | watermarked vs attacked | post-attack bit confidence
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='#0d1117')
        for ax in axes:
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#5a7080')
            ax.spines[:].set_color('#1e2730')

        n = len(density)
        x = np.arange(n)

        axes[0].plot(x, original_density, color='#4d9cff', lw=0.8, alpha=0.8)
        axes[0].set_title("Original Density", color='#c8d8e8')
        axes[0].set_xlabel("Element Index", color='#5a7080')
        axes[0].grid(True, alpha=0.1, color='#1e2730')

        axes[1].plot(x, density,  color='#39ff8a', lw=0.8, alpha=0.8, label='Watermarked')
        axes[1].plot(x, attacked, color='#ff6b35', lw=0.8, alpha=0.8, label='After Attack')
        axes[1].set_title(f"Attack: {attack_result['meta']['attack']}", color='#c8d8e8')
        axes[1].set_xlabel("Element Index", color='#5a7080')
        axes[1].legend(facecolor='#0f1318', edgecolor='#1e2730', labelcolor='#c8d8e8')
        axes[1].grid(True, alpha=0.1, color='#1e2730')

        conf   = detect_result['confidence']
        colors = ['#39ff8a' if c > 0.1 else '#ff6b35' for c in conf]
        axes[2].bar(range(len(conf)), conf, color=colors, width=0.8)
        axes[2].axhline(0.1, color='#00e5ff', linestyle='--', lw=1.5)
        score    = detect_result['correlation_score']
        detected = detect_result['is_watermarked']
        axes[2].set_title(
            f"Post-Attack Detection: {score}% ({'✓' if detected else '✗'})",
            color='#39ff8a' if detected else '#ff6b35'
        )
        axes[2].set_xlabel("Bit Index", color='#5a7080')
        axes[2].set_ylabel("Correlation", color='#5a7080')
        axes[2].grid(True, alpha=0.1, color='#1e2730')
        plt.tight_layout()

        return jsonify({
            'success':                     True,
            'attack_meta':                 attack_result['meta'],
            'is_watermarked_after_attack': detect_result['is_watermarked'],
            'correlation_score':           detect_result['correlation_score'],
            'detected_message':            detect_result['detected_message'],
            'image':                       fig_to_base64(fig),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── HEALTH CHECK ──────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Liveness check — visit http://127.0.0.1:5000/health to confirm backend is up."""
    return jsonify({'status': 'ok'})


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n  Hack3D backend → http://127.0.0.1:5000")
    print("  React frontend → http://localhost:3000\n")
    app.run(debug=True, port=5000, use_reloader=False, threaded=True)
