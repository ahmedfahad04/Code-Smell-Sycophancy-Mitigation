import streamlit as st
import json
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "mlcq_filtered.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# Set page config
st.set_page_config(
    page_title="Code Smell Analysis Comparison",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Code Smell Analysis Comparison Tool")
st.markdown("---")

# Cache dataset loading
@st.cache_data
def load_dataset():
    """Load the reference dataset from mlcq_filtered.json."""
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {item['id']: item for item in data}
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return {}

# Functions
def get_all_json_files(base_path: str) -> List[str]:
    """Get all JSON files from results directory and subdirectories."""
    json_files = []
    base = Path(base_path)
    
    # Get files from root
    for file in base.glob("*.json"):
        json_files.append(str(file))
    
    # Get files from subdirectories
    for subdir in base.glob("*/"):
        for file in subdir.glob("*.json"):
            json_files.append(str(file))
    
    return sorted(json_files)


def get_json_directories(base_path: Path) -> List[str]:
    """Return relative directory paths that contain at least one JSON file."""
    if not base_path.exists():
        return []

    directories = set()

    if any(base_path.glob("*.json")):
        directories.add(".")

    for subdir in base_path.rglob("*"):
        if subdir.is_dir() and any(subdir.glob("*.json")):
            directories.add(str(subdir.relative_to(base_path)))

    return sorted(directories)


def get_json_files_in_directory(base_path: Path, rel_dir: str) -> List[str]:
    """Return JSON files inside a selected relative directory."""
    selected_dir = base_path if rel_dir == "." else base_path / rel_dir
    if not selected_dir.exists() or not selected_dir.is_dir():
        return []
    return sorted(str(file) for file in selected_dir.glob("*.json"))

def load_json_file(filepath: str) -> List[Dict]:
    """Load JSON file and return data."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading {filepath}: {e}")
        return []

def get_file_display_name(filepath: str) -> str:
    """Get a clean display name for the file."""
    # Get relative path from results directory
    base = RESULTS_DIR
    try:
        rel_path = Path(filepath).relative_to(base)
        return str(rel_path)
    except:
        return Path(filepath).name

def compare_files(file1_data: List[Dict], file2_data: List[Dict], dataset: Dict) -> Tuple[pd.DataFrame, Dict]:
    """Compare two result files and find differences."""
    # Create dictionaries for quick lookup
    file1_dict = {item['id']: item for item in file1_data}
    file2_dict = {item['id']: item for item in file2_data}
    
    # Get all IDs
    all_ids = set(file1_dict.keys()) | set(file2_dict.keys())
    
    # Build comparison data
    differences = []
    severity_diff_count = 0
    
    for id_val in sorted(all_ids):
        file1_item = file1_dict.get(id_val)
        file2_item = file2_dict.get(id_val)
        dataset_item = dataset.get(id_val, {})
        
        file1_severity = file1_item['severity'] if file1_item else "N/A"
        file2_severity = file2_item['severity'] if file2_item else "N/A"
        dataset_severity = dataset_item.get('severity', 'N/A')
        smell_type = dataset_item.get('smell', 'N/A')
        code_snippet = dataset_item.get('code_snippet', 'N/A')
        
        has_difference = file1_severity != file2_severity
        if has_difference:
            severity_diff_count += 1
        
        differences.append({
            'ID': id_val,
            'Smell': smell_type,
            'Dataset Severity': dataset_severity,
            'File 1 Severity': file1_severity,
            'File 2 Severity': file2_severity,
            'Difference': '⚠️ YES' if has_difference else '✓ No',
            'File 1 Reasoning': file1_item['reasoning'] if file1_item else "N/A",
            'File 2 Reasoning': file2_item['reasoning'] if file2_item else "N/A",
            'Code Snippet': code_snippet,
        })
    
    df = pd.DataFrame(differences)
    
    stats = {
        'total_ids': len(all_ids),
        'common_ids': len(set(file1_dict.keys()) & set(file2_dict.keys())),
        'file1_only': len(set(file1_dict.keys()) - set(file2_dict.keys())),
        'file2_only': len(set(file2_dict.keys()) - set(file1_dict.keys())),
        'severity_differences': severity_diff_count,
    }
    
    return df, stats

# Load dataset once at startup
dataset = load_dataset()

# Initialize session state
if 'file1_path' not in st.session_state:
    st.session_state.file1_path = None
if 'file2_path' not in st.session_state:
    st.session_state.file2_path = None
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = None
if 'stats' not in st.session_state:
    st.session_state.stats = None

# Sidebar - File Selection
st.sidebar.header("📁 File Selection")

results_dir = RESULTS_DIR
json_files = get_all_json_files(str(results_dir))

if not json_files:
    st.error("No JSON files found in results directory!")
    st.stop()

json_directories = get_json_directories(results_dir)
if not json_directories:
    st.error("No directories with JSON files were found under results!")
    st.stop()

st.sidebar.markdown("**Select File 1:**")
file1_dir = st.sidebar.selectbox(
    "Choose directory for file 1",
    json_directories,
    key="file1_dir_select",
    format_func=lambda x: "results/" if x == "." else f"results/{x}"
)
file1_options = get_json_files_in_directory(results_dir, file1_dir)
if not file1_options:
    st.sidebar.error("No JSON files found in selected directory for File 1")
    st.stop()
file1_display_names = [get_file_display_name(f) for f in file1_options]
file1_display = st.sidebar.selectbox(
    "Choose first file",
    file1_display_names,
    key="file1_select"
)
file1_path = file1_options[file1_display_names.index(file1_display)]

st.sidebar.markdown("---")
st.sidebar.markdown("**Select File 2:**")
file2_dir = st.sidebar.selectbox(
    "Choose directory for file 2",
    json_directories,
    key="file2_dir_select",
    format_func=lambda x: "results/" if x == "." else f"results/{x}"
)
file2_options = get_json_files_in_directory(results_dir, file2_dir)
if not file2_options:
    st.sidebar.error("No JSON files found in selected directory for File 2")
    st.stop()
file2_display_names = [get_file_display_name(f) for f in file2_options]
file2_display = st.sidebar.selectbox(
    "Choose second file",
    file2_display_names,
    index=1 if len(file2_display_names) > 1 else 0,
    key="file2_select"
)
file2_path = file2_options[file2_display_names.index(file2_display)]

st.sidebar.markdown("---")

# Load and compare only when button is clicked
if st.sidebar.button("🔍 Compare Files", type="primary", use_container_width=True):
    with st.spinner("Loading and comparing files..."):
        file1_data = load_json_file(file1_path)
        file2_data = load_json_file(file2_path)
        
        if file1_data and file2_data:
            df_comparison, stats = compare_files(file1_data, file2_data, dataset)
            st.session_state.comparison_data = df_comparison
            st.session_state.stats = stats
            st.session_state.file1_path = file1_path
            st.session_state.file2_path = file2_path

# Display only if comparison data is available
if st.session_state.comparison_data is not None:
    df_comparison = st.session_state.comparison_data
    stats = st.session_state.stats
    
    st.markdown(f"### 📊 Comparison Results")
    st.markdown(f"**File 1:** {get_file_display_name(st.session_state.file1_path)}")
    st.markdown(f"**File 2:** {get_file_display_name(st.session_state.file2_path)}")
    st.markdown("---")
    
    # Display Statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total IDs", stats['total_ids'])
    with col2:
        st.metric("Common IDs", stats['common_ids'])
    with col3:
        st.metric("File 1 Only", stats['file1_only'])
    with col4:
        st.metric("File 2 Only", stats['file2_only'])
    with col5:
        st.metric("Severity Diffs", f"**{stats['severity_differences']}**")
    
    st.markdown("---")
    
    # Filters
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        show_differences_only = st.checkbox("Show only differences", value=True)
    
    with col_filter2:
        id_filter = st.text_input("Filter by ID (optional):", placeholder="e.g., 125")
    
    with col_filter3:
        severity_filter = st.multiselect(
            "Filter by severity difference:",
            ["⚠️ YES", "✓ No"],
            default=["⚠️ YES"]
        )
    
    # Apply filters
    filtered_df = df_comparison.copy()
    
    if id_filter:
        try:
            id_val = int(id_filter)
            filtered_df = filtered_df[filtered_df['ID'] == id_val]
        except ValueError:
            st.warning("Please enter a valid ID number")
    
    if severity_filter:
        filtered_df = filtered_df[filtered_df['Difference'].isin(severity_filter)]
    
    st.markdown("---")
    
    # Display results
    if len(filtered_df) == 0:
        st.info("No results match your filters")
    else:
        st.markdown(f"**Showing {len(filtered_df)} record(s)**")
        
        # Display each comparison
        for idx, row in filtered_df.iterrows():
            # Header with ID and smell
            header_color = "🔴" if row['Difference'] == "⚠️ YES" else "✅"
            
            col1, col2, col3, col4, col5, col6 = st.columns([1, 1.5, 1, 1, 1, 1])
            
            with col1:
                st.markdown(f"### {header_color} ID {int(row['ID'])}")
            
            with col2:
                st.markdown(f"**Smell:** `{row['Smell']}`")
            
            with col3:
                severity_badge = "🟢" if row['Dataset Severity'] == "none" else "🟡" if row['Dataset Severity'] == "minor" else "🔴"
                st.markdown(f"**Dataset:** {severity_badge} `{row['Dataset Severity']}`")
            
            with col4:
                sev1_badge = "🟢" if row['File 1 Severity'] == "none" else "🟡" if row['File 1 Severity'] == "minor" else "🔴"
                st.markdown(f"**File 1:** {sev1_badge} `{row['File 1 Severity']}`")
            
            with col5:
                sev2_badge = "🟢" if row['File 2 Severity'] == "none" else "🟡" if row['File 2 Severity'] == "minor" else "🔴"
                st.markdown(f"**File 2:** {sev2_badge} `{row['File 2 Severity']}`")
            
            with col6:
                st.markdown(f"**Diff:** {row['Difference']}")
            
            # Reasoning comparison (side by side)
            st.markdown("**Reasoning:**")
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                st.markdown(f"*File 1 ({get_file_display_name(st.session_state.file1_path).split('/')[-1]})*")
                st.info(row['File 1 Reasoning'])
            
            with col_r2:
                st.markdown(f"*File 2 ({get_file_display_name(st.session_state.file2_path).split('/')[-1]})*")
                st.info(row['File 2 Reasoning'])
            
            # Expandable code section
            with st.expander(f"📄 Code Snippet (ID {int(row['ID'])})"):
                if row['Code Snippet'] != "N/A":
                    st.code(row['Code Snippet'], language="java")
                else:
                    st.info("Code snippet not available in dataset")
            
            st.markdown("---")
        
        # Statistics table
        st.markdown("## 📈 Severity Differences Summary")
        
        diff_df = filtered_df[filtered_df['Difference'] == "⚠️ YES"].copy()
        
        if len(diff_df) > 0:
            summary_data = []
            
            for idx, row in diff_df.iterrows():
                summary_data.append({
                    'ID': int(row['ID']),
                    'Smell': row['Smell'],
                    'Dataset': row['Dataset Severity'],
                    'File 1': row['File 1 Severity'],
                    'File 2': row['File 2 Severity'],
                    'Change': f"{row['File 1 Severity']} → {row['File 2 Severity']}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Export option
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Differences as CSV",
                data=csv,
                file_name="severity_differences.csv",
                mime="text/csv"
            )
        else:
            st.info("No severity differences to show with current filters")

else:
    st.info("👈 Select two files from the sidebar and click 'Compare Files' to start analyzing")
