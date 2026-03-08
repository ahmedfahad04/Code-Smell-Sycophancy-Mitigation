import pandas as pd
import random
import plotly.graph_objects as go

# ==========================================
# 1. GENERATE REALISTIC MULTI-BIAS DUMMY DATA
# ==========================================
random.seed(42)
dummy_records = []

# Simulate 100 code snippets evaluated across all strategies
for i in range(1, 101):
    # The Ground Truth Baseline
    casual = random.choices(['none', 'minor', 'major', 'critical'], weights=[20, 40, 30, 10])[0]
    
    # --- Attack Stage (The 3 Biases) ---
    # Confirmation-Bias: High sycophancy (70% DFR)
    cb = 'none' if casual != 'none' and random.random() < 0.70 else casual
    
    # False-Premise: Highest sycophancy due to fake tool authority (90% DFR)
    fp = 'none' if casual != 'none' and random.random() < 0.90 else casual
    
    # Contradictory-Hint: Moderate sycophancy but causes confusion (60% DFR)
    ch = 'none' if casual != 'none' and random.random() < 0.60 else casual

    # --- Cure Stage (Evidence-Based CoT) ---
    # The prompt almost entirely restores the ground truth (95% Recovery)
    cot_cb = casual if random.random() < 0.95 else cb
    cot_fp = casual if random.random() < 0.95 else fp
    cot_ch = casual if random.random() < 0.95 else ch

    # Append separate flow records for each attack vector
    dummy_records.append({'id': i, 'Strategy': 'CB', 'Casual': casual, 'Bias_Result': cb, 'CoT_Result': cot_cb})
    dummy_records.append({'id': i, 'Strategy': 'FP', 'Casual': casual, 'Bias_Result': fp, 'CoT_Result': cot_fp})
    dummy_records.append({'id': i, 'Strategy': 'CH', 'Casual': casual, 'Bias_Result': ch, 'CoT_Result': cot_ch})

df = pd.DataFrame(dummy_records)

# ==========================================
# 2. PREPARE SANKEY DIAGRAM NODES & FLOWS
# ==========================================
severities = ['none', 'minor', 'major', 'critical']
nodes = []

# Create strict node categories to force the 3-pillar layout
for sev in severities: nodes.append(f"Casual: {sev}")
for sev in severities: nodes.append(f"CB: {sev}")
for sev in severities: nodes.append(f"FP: {sev}")
for sev in severities: nodes.append(f"CH: {sev}")
for sev in severities: nodes.append(f"CoT: {sev}")

node_dict = {node: i for i, node in enumerate(nodes)}

# Define colors (Green=None, Yellow=Minor, Orange=Major, Red=Critical)
color_map = {"none": "#2ecc71", "minor": "#f1c40f", "major": "#e67e22", "critical": "#e74c3c"}
node_colors = [color_map[n.split(": ")[1]] for n in nodes]

# Format clean labels for the chart
display_labels = [n.replace("CB:", "Conf-Bias:")
                   .replace("FP:", "False-Premise:")
                   .replace("CH:", "Contra-Hint:") for n in nodes]

source, target, value = [], [], []

# Flow 1: Casual Baseline -> The 3 Bias Attacks
for strategy in ['CB', 'FP', 'CH']:
    counts = df[df['Strategy'] == strategy].groupby(['Casual', 'Bias_Result']).size().reset_index(name='count')
    for _, row in counts.iterrows():
        source.append(node_dict[f"Casual: {row['Casual']}"])
        target.append(node_dict[f"{strategy}: {row['Bias_Result']}"])
        value.append(row['count'])

# Flow 2: The 3 Bias Attacks -> Evidence-Based CoT (The Cure)
for strategy in ['CB', 'FP', 'CH']:
    counts = df[df['Strategy'] == strategy].groupby(['Bias_Result', 'CoT_Result']).size().reset_index(name='count')
    for _, row in counts.iterrows():
        source.append(node_dict[f"{strategy}: {row['Bias_Result']}"])
        target.append(node_dict[f"CoT: {row['CoT_Result']}"])
        value.append(row['count'])

# ==========================================
# 3. BUILD AND SHOW THE PLOTLY FIGURE
# ==========================================
fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=display_labels,
        color=node_colors
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color="rgba(189, 195, 199, 0.3)"  # Transparent grey to prevent visual clutter
    )
)])

# Add layout styling
fig.update_layout(
    title_text="<b>Multi-Vector Sycophancy Degradation and Recovery Flow</b><br>Tracking Ground Truth Across Three Distinct Bias Attacks",
    font_size=12,
    width=1200,
    height=800,
    margin=dict(t=80, b=40, l=40, r=40)
)

# Open in browser
fig.show()

# fig.write_image("multi_bias_sankey.png", scale=3)