from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
CORS(app, resources={r"/*": {"origins": "*"}})


# ── SHARED STATE (per-run, single-user dev server) ────────────────────────────
_run_state = {
    "running": False,
    "progress": [],       # list of dicts, one per iteration
    "result": None,
    "error": None,
}
_state_lock = threading.Lock()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64


def plot_3d_design(nodes, elems, density, threshold=0.5, title="3D Structure"):
    fig = plt.figure(figsize=(10, 7), facecolor='#0d1117')
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor('#0d1117')
    mask = density > threshold
    active_elems = np.where(mask)[0]
    if len(active_elems) == 0:
        ax.text(0.5, 0.5, 0.5, f"No elements with density > {threshold}",
                transform=ax.transAxes, ha="center", fontsize=12, color='#c8d8e8')
        ax.set_title(title, color='#c8d8e8')
        return fig
    cmap = matplotlib.colormaps["RdYlGn"]
    norm = Normalize(vmin=threshold, vmax=1.0)
    for elem_idx in active_elems:
        elem_nodes = elems[elem_idx]
        elem_coords = nodes[elem_nodes]
        elem_density = density[elem_idx]
        color = cmap(norm(elem_density))
        faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]
        for face in faces:
            ax.add_collection3d(Poly3DCollection(
                [elem_coords[face]], facecolors=color, edgecolors='k', linewidths=0.3
            ))
    all_coords = nodes[elems[active_elems]].reshape(-1, 3)
    ax.set_xlim([all_coords[:,0].min(), all_coords[:,0].max()])
    ax.set_ylim([all_coords[:,1].min(), all_coords[:,1].max()])
    ax.set_zlim([all_coords[:,2].min(), all_coords[:,2].max()])
    ax.set_box_aspect([np.ptp(all_coords[:,0]), np.ptp(all_coords[:,1]), np.ptp(all_coords[:,2])])
    ax.set_xlabel("X (m)", color='#5a7080'); ax.set_ylabel("Y (m)", color='#5a7080'); ax.set_zlabel("Z (m)", color='#5a7080')
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
    """Build and configure a FEM solver from request data."""
    nx          = int(data.get('nx', 20))
    ny          = int(data.get('ny', 6))
    nz          = int(data.get('nz', 4))
    fixed_face  = data.get('fixedFace', 'x0')
    load_face   = data.get('loadFace', 'x1')
    load_dir    = int(data.get('loadDirection', -1))
    load_mag    = float(data.get('loadMagnitude', 1e4))

    fem = HexFEMSolver3D(E_mod=200e9, nu=0.3)
    fem.set_mesh(Lx=1.0, Ly=0.2, Lz=0.1, nx=nx, ny=ny, nz=nz)

    face_map = {
        'x0': (0, 0.0), 'x1': (0, 1.0),
        'y0': (1, 0.0), 'y1': (1, 0.2),
        'z0': (2, 0.0), 'z1': (2, 0.1),
    }
    bc_axis, bc_coord = face_map.get(fixed_face, (0, 0.0))
    fem.fix_face(axis=bc_axis, coord=bc_coord)

    load_axis, load_coord = face_map.get(load_face, (0, 1.0))
    fem.add_distributed_load(
        axis=load_axis, coord=load_coord,
        direction=load_dir,
        total=load_mag / (fem.ny * fem.nz)
    )
    return fem


def build_plots(fem, density, volume_fraction, history):
    """Generate all matplotlib figures and return as base64 strings."""
    # Convergence
    fig_conv, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='#0d1117')
    for ax in axes:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#5a7080')
        ax.spines[:].set_color('#1e2730')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color('#5a7080')

    iters = history['iteration']
    axes[0].semilogy(iters, history['compliance'], color='#4d9cff', lw=2, marker='o', ms=3)
    axes[0].set_title("Compliance Convergence", color='#c8d8e8'); axes[0].set_xlabel("Iteration", color='#5a7080'); axes[0].grid(True, alpha=0.15, color='#1e2730')

    axes[1].plot(iters, history['volume'], color='#39ff8a', lw=2, marker='o', ms=3, label='Actual')
    axes[1].axhline(volume_fraction, color='#ff6b35', linestyle='--', lw=2, label=f'Target ({volume_fraction:.2f})')
    axes[1].set_title("Volume Constraint", color='#c8d8e8'); axes[1].set_xlabel("Iteration", color='#5a7080'); axes[1].legend(facecolor='#0f1318', edgecolor='#1e2730', labelcolor='#c8d8e8'); axes[1].grid(True, alpha=0.15, color='#1e2730')

    axes[2].semilogy(iters, history['density_change'], color='#ff9f43', lw=2, marker='o', ms=3)
    axes[2].set_title("Convergence Indicator", color='#c8d8e8'); axes[2].set_xlabel("Iteration", color='#5a7080'); axes[2].grid(True, alpha=0.15, color='#1e2730')

    plt.tight_layout()
    conv_img = fig_to_base64(fig_conv)

    # 3D structures
    structure_imgs = {}
    for t in [0.1, 0.3, 0.5]:
        fig3d = plot_3d_design(fem.nodes_np, fem.elems_t, density, threshold=t,
                               title=f"Optimized design (density > {t})")
        structure_imgs[str(t)] = fig_to_base64(fig3d)

    # Histogram
    fig_hist, ax = plt.subplots(figsize=(8, 5), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#5a7080'); ax.spines[:].set_color('#1e2730')
    ax.hist(density, bins=30, color='#4d9cff', edgecolor='#0d1117', alpha=0.8)
    ax.axvline(np.mean(density), color='#ff6b35', linestyle='--', lw=2, label=f'Mean: {np.mean(density):.3f}')
    ax.axvline(volume_fraction, color='#39ff8a', linestyle='--', lw=2, label=f'Target: {volume_fraction:.3f}')
    ax.set_xlabel("Density", color='#5a7080'); ax.set_ylabel("Elements", color='#5a7080')
    ax.set_title("Density Distribution", color='#c8d8e8')
    ax.legend(facecolor='#0f1318', edgecolor='#1e2730', labelcolor='#c8d8e8')
    ax.grid(True, alpha=0.15, color='#1e2730')
    plt.tight_layout()
    hist_img = fig_to_base64(fig_hist)

    return conv_img, structure_imgs, hist_img


# ── SSE OPTIMIZE ENDPOINT ─────────────────────────────────────────────────────

@app.route('/optimize/stream', methods=['POST'])
def optimize_stream():
    """
    SSE endpoint — streams per-iteration progress as JSON events,
    then sends a final 'done' event with all images.
    """
    data = request.json

    volume_frac = float(data.get('volumeFraction', 0.2))
    penalty     = float(data.get('penalty', 3.0))
    iterations  = int(data.get('iterations', 30))

    def generate():
        try:
            # Build FEM
            yield f"data: {json.dumps({'type':'status','msg':'Building FEM mesh…'})}\n\n"
            fem = build_fem(data)

            yield f"data: {json.dumps({'type':'status','msg':'Setting up SIMP optimizer…'})}\n\n"
            optimizer = SIMPOptimizer(
                fem_solver=fem,
                initial_density=volume_frac,
                volume_fraction=volume_frac,
                penalty=penalty,
                filter_radius=0.02,
            )

            yield f"data: {json.dumps({'type':'status','msg':'Running optimization…'})}\n\n"

            # Run iteration by iteration so we can stream
            for iteration in range(iterations):
                results     = fem.solve(optimizer.density)
                compliance  = results["compliance"]
                sensitivities = results["sensitivities"]

                density_new    = optimizer.update_density(sensitivities)
                density_change = float(np.max(np.abs(density_new - optimizer.density)))
                optimizer.density = density_new

                volume = float(np.sum(optimizer.density) / optimizer.n_elem)
                optimizer.history["compliance"].append(compliance)
                optimizer.history["volume"].append(volume)
                optimizer.history["density_change"].append(density_change)
                optimizer.history["iteration"].append(iteration)

                # Stream iteration update
                payload = {
                    "type":      "iteration",
                    "iteration": iteration,
                    "total":     iterations,
                    "compliance": round(float(compliance), 8),
                    "volume":    round(volume, 5),
                    "density_change": round(density_change, 6),
                    "pct":       round((iteration + 1) / iterations * 100, 1),
                }
                yield f"data: {json.dumps(payload)}\n\n"

            # Build plots
            yield f"data: {json.dumps({'type':'status','msg':'Generating visualizations…'})}\n\n"
            density  = optimizer.density
            history  = optimizer.history
            conv_img, structure_imgs, hist_img = build_plots(fem, density, volume_frac, history)

            final = {
                "type": "done",
                "metrics": {
                    "finalCompliance": round(float(history['compliance'][-1]), 8),
                    "finalVolume":     round(float(history['volume'][-1]), 4),
                    "iterations":      iterations,
                },
                "images": {
                    "convergence": conv_img,
                    "histogram":   hist_img,
                    "structure":   structure_imgs,
                },
                # pass density so watermark tab can use it
                "density": density.tolist(),
            }
            yield f"data: {json.dumps(final)}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type':'error','msg':str(e),'trace':traceback.format_exc()})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Access-Control-Allow-Origin': '*',
        }
    )


# ── WATERMARK ENDPOINTS ───────────────────────────────────────────────────────

@app.route('/watermark/embed', methods=['POST'])
def watermark_embed():
    """Embed a watermark into a density field and return comparison images."""
    try:
        data       = request.json
        density    = np.array(data['density'], dtype=float)
        message    = data.get('message', 'NYU-HACK3D')
        alpha      = float(data.get('alpha', 0.03))
        secret_key = data.get('secretKey', 'hack3d-nyu-vip-2025')

        wm = DensityWatermark(secret_key=secret_key, alpha=alpha)
        result = wm.embed(density, message=message)

        # Perturbation heatmap
        perturbation = np.array(result['perturbation'])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='#0d1117')
        for ax in axes:
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#5a7080')
            ax.spines[:].set_color('#1e2730')

        n = len(density)
        x = np.arange(n)
        axes[0].bar(x, density, color='#4d9cff', alpha=0.6, width=1.0, label='Original')
        axes[0].bar(x, result['watermarked_density'], color='#39ff8a', alpha=0.4, width=1.0, label='Watermarked')
        axes[0].set_title("Density Comparison", color='#c8d8e8')
        axes[0].set_xlabel("Element Index", color='#5a7080')
        axes[0].legend(facecolor='#0f1318', edgecolor='#1e2730', labelcolor='#c8d8e8')
        axes[0].grid(True, alpha=0.1, color='#1e2730')

        axes[1].plot(x, perturbation, color='#ff6b35', lw=0.8, alpha=0.9)
        axes[1].axhline(0, color='#1e2730', lw=1)
        axes[1].fill_between(x, perturbation, 0, where=(perturbation > 0), color='#39ff8a', alpha=0.3)
        axes[1].fill_between(x, perturbation, 0, where=(perturbation < 0), color='#ff6b35', alpha=0.3)
        axes[1].set_title(f"Watermark Signal (α={alpha})", color='#c8d8e8')
        axes[1].set_xlabel("Element Index", color='#5a7080')
        axes[1].set_ylabel("Perturbation", color='#5a7080')
        axes[1].grid(True, alpha=0.1, color='#1e2730')

        plt.tight_layout()
        embed_img = fig_to_base64(fig)

        return jsonify({
            'success': True,
            'watermarked_density': result['watermarked_density'].tolist(),
            'snr_db': result['snr_db'],
            'alpha': alpha,
            'message': message,
            'n_bits': result['n_bits'],
            'image': embed_img,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/watermark/detect', methods=['POST'])
def watermark_detect():
    """Detect watermark in a (possibly attacked) density field."""
    try:
        data             = request.json
        density          = np.array(data['density'], dtype=float)
        original_density = np.array(data.get('original_density', data['density']), dtype=float)
        secret_key       = data.get('secretKey', 'hack3d-nyu-vip-2025')
        n_bits           = int(data.get('n_bits', 64))

        wm = DensityWatermark(secret_key=secret_key)
        result = wm.detect(density, original=original_density, n_bits=n_bits)

        # Confidence bar chart
        conf = result['confidence']
        fig, ax = plt.subplots(figsize=(12, 3), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#5a7080')
        ax.spines[:].set_color('#1e2730')

        colors = ['#39ff8a' if c > 0.1 else '#ff6b35' for c in conf]
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
        detect_img = fig_to_base64(fig)

        return jsonify({
            'success': True,
            'is_watermarked': result['is_watermarked'],
            'detected_message': result['detected_message'],
            'correlation_score': result['correlation_score'],
            'avg_confidence': result['avg_confidence'],
            'image': detect_img,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/watermark/attack', methods=['POST'])
def watermark_attack():
    """Simulate an adversarial attack then re-detect the watermark."""
    try:
        data             = request.json
        density          = np.array(data['density'], dtype=float)
        original_density = np.array(data['original_density'], dtype=float)
        attack           = data.get('attack', 'noise')
        secret_key       = data.get('secretKey', 'hack3d-nyu-vip-2025')

        attack_params = {
            'sigma':    float(data.get('sigma', 0.05)),
            'factor':   float(data.get('factor', 0.9)),
            'fraction': float(data.get('fraction', 0.2)),
            'n_levels': int(data.get('n_levels', 5)),
            'window':   int(data.get('window', 5)),
        }

        wm = DensityWatermark(secret_key=secret_key)
        attack_result = wm.simulate_attack(density, attack=attack, **attack_params)
        attacked = attack_result['attacked_density']

        detect_result = wm.detect(attacked, original=original_density)

        # Side-by-side comparison plot
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

        axes[1].plot(x, density, color='#39ff8a', lw=0.8, alpha=0.8, label='Watermarked')
        axes[1].plot(x, attacked, color='#ff6b35', lw=0.8, alpha=0.8, label='After Attack')
        axes[1].set_title(f"Attack: {attack_result['meta']['attack']}", color='#c8d8e8')
        axes[1].set_xlabel("Element Index", color='#5a7080')
        axes[1].legend(facecolor='#0f1318', edgecolor='#1e2730', labelcolor='#c8d8e8')
        axes[1].grid(True, alpha=0.1, color='#1e2730')

        conf = detect_result['confidence']
        colors = ['#39ff8a' if c > 0.1 else '#ff6b35' for c in conf]
        axes[2].bar(range(len(conf)), conf, color=colors, width=0.8)
        axes[2].axhline(0.1, color='#00e5ff', linestyle='--', lw=1.5)
        score = detect_result['correlation_score']
        detected = detect_result['is_watermarked']
        axes[2].set_title(
            f"Post-Attack Detection: {score}% ({'✓' if detected else '✗'})",
            color='#39ff8a' if detected else '#ff6b35'
        )
        axes[2].set_xlabel("Bit Index", color='#5a7080')
        axes[2].set_ylabel("Correlation", color='#5a7080')
        axes[2].grid(True, alpha=0.1, color='#1e2730')

        plt.tight_layout()
        attack_img = fig_to_base64(fig)

        return jsonify({
            'success': True,
            'attack_meta': attack_result['meta'],
            'is_watermarked_after_attack': detect_result['is_watermarked'],
            'correlation_score': detect_result['correlation_score'],
            'detected_message': detect_result['detected_message'],
            'image': attack_img,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── HEALTH ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False, threaded=True)