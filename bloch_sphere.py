import numpy as np
import plotly.graph_objects as go
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# ---- USER INPUT SECTION ----
theta_deg = float(input("Enter final rotation angle θ (in degrees): "))
rotation_gate = input("Enter rotation gate (Rx, Ry, Rz): ").strip().lower()
theta = np.radians(theta_deg)

# Initial normalized state
alpha = np.sqrt(3) / 2
beta = complex(0.25, np.sqrt(3) / 4)
norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
alpha /= norm
beta /= norm

# Get Bloch vector, θ (elevation from Z), φ (azimuth from X)
def get_bloch_vector_and_angles(statevector):
    a, b = statevector.data
    x = 2 * (a.conjugate() * b).real
    y = 2 * (a.conjugate() * b).imag
    z = abs(a)*2 - abs(b)*2
    r = np.sqrt(x*2 + y*2 + z*2)
    theta_rad = np.arccos(z / r)
    phi_rad = np.arctan2(y, x)
    theta_deg = np.degrees(theta_rad)
    phi_deg = np.degrees(phi_rad) % 360
    return [x, y, z, theta_deg, phi_deg]

# Animation vectors
thetas = np.linspace(0, theta, 60)
vectors = []

for t in thetas:
    qc = QuantumCircuit(1)
    qc.initialize([alpha, beta], 0)
    if rotation_gate == 'rx':
        qc.rx(t, 0)
    elif rotation_gate == 'ry':
        qc.ry(t, 0)
    elif rotation_gate == 'rz':
        qc.rz(t, 0)
    else:
        raise ValueError("Rotation gate must be Rx, Ry, or Rz")
    state = Statevector.from_instruction(qc)
    vectors.append(get_bloch_vector_and_angles(state))

# Sphere mesh
u, v = np.mgrid[0:np.pi:100j, 0:2*np.pi:100j]
x_sphere = np.sin(u) * np.cos(v)
y_sphere = np.sin(u) * np.sin(v)
z_sphere = np.cos(u)

fig = go.Figure()

# Bloch sphere surface
fig.add_trace(go.Surface(
    x=x_sphere, y=y_sphere, z=z_sphere,
    opacity=0.2, showscale=False,
    colorscale='Blues', hoverinfo='skip'
))

# Axes (X=pink, Y=green, Z=blue)
axes = {
    "X": ([0, 1.2], [0, 0], [0, 0], 'deeppink'),
    "Y": ([0, 0], [0, 1.2], [0, 0], 'green'),
    "Z": ([0, 0], [0, 0], [0, 1.2], 'blue')
}
for name, (x, y, z, color) in axes.items():
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color=color, width=6),
        name=f"{name}-axis"
    ))

# Initial vector
x0, y0, z0, theta0, phi0 = vectors[0]
fig.add_trace(go.Scatter3d(
    x=[0, x0], y=[0, y0], z=[0, z0],
    mode='lines+markers',
    line=dict(color='crimson', width=8),
    marker=dict(size=4),
    name="Bloch Vector"
))

# Initial label
# fig.add_trace(go.Scatter3d(
#     x=[0], y=[0], z=[1.4],
#     mode='text',
#     text=[f"x={x0:.2f}, y={y0:.2f}, z={z0:.2f}<br>θ={theta0:.1f}°, φ={phi0:.1f}°"],
#     textposition="top center",
#     showlegend=False
# ))

# ---- Animation Frames ----
frames = []
for i, (x, y, z, theta_d, phi_d) in enumerate(vectors):
    frames.append(go.Frame(
        data=[
            go.Scatter3d(
                x=[0, x], y=[0, y], z=[0, z],
                mode='lines+markers',
                line=dict(color='crimson', width=8),
                marker=dict(size=4)
            ),
            go.Scatter3d(
                x=[0], y=[0], z=[1.4],
                mode='text',
                text=[f"x={x:.2f}, y={y:.2f}, z={z:.2f}<br>θ={theta_d:.1f}°, φ={phi_d:.1f}°"],
                textfont=dict(size=30, color="black")
            )
        ],
        name=str(i)
    ))
fig.frames = frames

# ---- Layout with Controls ----
fig.update_layout(
    title=f"Bloch Sphere: {rotation_gate.upper()} Rotation (θ = {theta_deg:.1f}°)",
    scene=dict(
        xaxis=dict(title='X', range=[-1.2, 1.2]),
        yaxis=dict(title='Y', range=[-1.2, 1.2]),
        zaxis=dict(title='Z', range=[-1.2, 1.6]),
        aspectmode='cube'
    ),
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        buttons=[
            dict(label="Play", method="animate",
                 args=[None, {"frame": {"duration": 80}, "fromcurrent": True}]),
            dict(label="Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
        ],
        x=0.1, y=1.15, xanchor="left", yanchor="top"
    )],
    sliders=[dict(
        steps=[dict(method="animate",
                    args=[[str(i)], {"frame": {"duration": 0}, "mode": "immediate"}],
                    label=str(i)) for i in range(len(vectors))],
        currentvalue={"prefix": "Step: "},
        pad={"t": 50}
    )]
)

fig.show()