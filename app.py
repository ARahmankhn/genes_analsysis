import io
import pandas as pd
import streamlit as st
import GEOparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import gseapy  # For GO enrichment analysis

# ---------- Helper Functions ----------
def normalize_data(df):
    """Z-score normalization."""
    return (df - df.mean(axis=0)) / df.std(axis=0)

def find_common_genes(*dfs):
    """Find common genes across datasets."""
    common_genes = dfs[0].index
    for df in dfs[1:]:
        common_genes = common_genes.intersection(df.index)
    return common_genes

def csv_download(df: pd.DataFrame, filename: str):
    """Download the DataFrame as a CSV file."""
    buf = io.StringIO()
    df.to_csv(buf, index=True)
    return buf.getvalue().encode("utf-8")

def differential_expression_analysis(df, group_labels):
    """Differential Expression using t-test between two groups."""
    unique_groups = set(group_labels)
    group_1_data = df.loc[:, group_labels == unique_groups.pop()]
    group_2_data = df.loc[:, group_labels == unique_groups.pop()]
    p_values = []
    for gene in df.index:
        _, p_value = ttest_ind(group_1_data.loc[gene], group_2_data.loc[gene])
        p_values.append(p_value)
    return pd.Series(p_values, index=df.index)

def go_enrichment_analysis(gene_list, organism="Human"):
    """Perform Gene Ontology (GO) enrichment using gseapy."""
    enrichr = gseapy.enrichr(gene_list=gene_list, gene_sets='GO_Biological_Process_2018', organism=organism)
    return enrichr.results

# ---------- Page & Styles ----------
st.set_page_config(page_title="GEO (GSE) Explorer", page_icon="🧬", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
h1, h2, h3 {font-weight: 800;}
.badge{
  display:inline-block;padding:.15rem .5rem;border-radius:.5rem;background:#eef5ff;
  border:1px solid #cfe1ff;font-size:.8rem;margin-right:.35rem
}
.info-box{
  background:#f8fafc;border:1px solid #e5e7eb;border-radius:.75rem;padding:14px;margin:.25rem 0;
}
.kpi {background:#f1f5f9;border:1px solid #e2e8f0;border-radius:.75rem;padding:12px;text-align:center}
.kpi h3{margin:0;font-size:.9rem;color:#475569}
.kpi p{margin:.25rem 0 0 0;font-size:1.05rem;font-weight:700}
.card{border:1px solid #e5e7eb;border-radius:1rem;padding:1rem;margin-bottom:1.25rem;background:white}
.hr{border-top:1px solid #e5e7eb;margin:1rem 0}
</style>
""", unsafe_allow_html=True)

st.title("🧬 GEO (GSE) Explorer — Multi-Dataset")
st.caption("Enter one or more **GSE accessions**. Fetch metadata, platforms, samples, and expression matrices on demand.")

# ---------- Inputs (dynamic) ----------
if "gse_fields" not in st.session_state:
    st.session_state.gse_fields = ["GSE114083", "GSE68849"]  # initial examples

st.markdown("### 🔢 Add GSE accessions")
btns1 = st.columns([1, 1, 1, 1, 2])
with btns1[0]:
    if st.button("➕ Add GSE input", use_container_width=True):
        if len(st.session_state.gse_fields) < 20:
            st.session_state.gse_fields.append("")
with btns1[1]:
    if st.button("➖ Remove last", use_container_width=True):
        if len(st.session_state.gse_fields) > 1:
            st.session_state.gse_fields.pop()
with btns1[2]:
    clear_inputs = st.button("🧹 Clear inputs", use_container_width=True)
with btns1[3]:
    clear_cache = st.button("♻️ Clear cache", use_container_width=True)

if clear_inputs:
    st.session_state.gse_fields = [""]

if clear_cache:
    st.cache_data.clear()
    st.success("Cache cleared.")

# Render dynamic inputs
new_vals = []
for i, val in enumerate(st.session_state.gse_fields):
    new_vals.append(st.text_input(f"GSE #{i+1}", value=val, key=f"gse_{i}", placeholder="e.g. GSE12345"))
st.session_state.gse_fields = new_vals

# Fetch Button
fetch_button = st.button("🔎 Fetch All GSEs", use_container_width=True)

def get_selected_accessions():
    accs = [a.strip() for a in st.session_state.gse_fields if a.strip()]
    if not accs:
        st.warning("Please enter at least one GSE accession.")
        return []
    return accs

# ---------- Fetch GSE Data Using GEOparse ----------
@st.cache_data(show_spinner=False)
def load_gse(accession: str):
    """Download and parse a GSE dataset directly from GEO and return the expression matrix and metadata."""
    try:
        # Download and parse the GSE data using GEOparse
        gse = GEOparse.get_GEO(geo=accession.strip(), destdir="./geo_cache", silent=True)

        # Extract metadata for the GSE
        title = gse.metadata.get("title", "Not available")
        summary = gse.metadata.get("summary", "Not available")
        submission_date = gse.metadata.get("submission_date", "Unknown")
        platforms = list(gse.gpls.keys())
        n_samples = len(gse.gsms)

        # Extract organism info from GSMs
        orgs = set()
        for gsm in gse.gsms.values():
            for key in ("organism_ch1", "organism", "source_name_ch1"):
                if key in gsm.metadata and len(gsm.metadata[key]) > 0:
                    orgs.add(str(gsm.metadata[key][0]))
                    break
        organisms = sorted(orgs) if orgs else ["Unknown"]

        # Extract gene expression matrix (assuming it's a microarray platform)
        expression_matrix = gse.pivot_samples("VALUE")  # For microarrays, "VALUE" is the typical column
        expression_matrix = expression_matrix.fillna(0)  # Fill missing values with 0 (or handle appropriately)

        # Per-platform feature counts (rows in platform table)
        feature_counts = {}
        for gpl_id, gpl in gse.gpls.items():
            try:
                feature_counts[gpl_id] = int(len(gpl.table))
            except Exception:
                feature_counts[gpl_id] = None

        return {
            "gse": gse,
            "title": title,
            "summary": summary,
            "date": submission_date,
            "platforms": platforms,
            "n_samples": n_samples,
            "organisms": organisms,
            "feature_counts": feature_counts,
            "expression_matrix": expression_matrix
        }

    except Exception as e:
        st.error(f"Error fetching GSE data for {accession}: {str(e)}")
        return None

# ---------- Fetch & Process GSEs ----------
if fetch_button:
    accessions = get_selected_accessions()
    if accessions:
        # Fetch data for each GSE and aggregate it
        st.subheader("📚 Datasets Overview")
        all_results = []
        all_dataframes = []  # To store all expression matrices for common gene identification
        for acc in accessions:
            try:
                info = load_gse(acc)
                gse = info["gse"]

                # Prepare the data for the table
                row = {
                    "Accession": acc,
                    "Organism": ", ".join(info["organisms"]),
                    "Platform ID": ", ".join(info["platforms"]),
                    "Probe Count": ", ".join([str(info["feature_counts"].get(gpl, "Unknown")) for gpl in info["platforms"]]),
                    "Submission Date": info["date"]
                }
                all_results.append(row)

                # Append the expression matrix for common gene identification
                all_dataframes.append(info["expression_matrix"])

            except Exception as e:
                st.error(f"❌ {acc}: {e}")

        # Check if the dataframes are not empty before proceeding
        if all_dataframes:
            common_genes = find_common_genes(*all_dataframes)
            st.write(f"Found {len(common_genes)} common genes across datasets.")

            # Filter each DataFrame to keep only the common genes
            common_dataframes = [df.loc[common_genes] for df in all_dataframes]

            # Normalize data (optional)
            normalized_dataframes = [normalize_data(df) for df in common_dataframes]

            # Combine the normalized data into one DataFrame for PCA
            combined_data = pd.concat(normalized_dataframes, axis=1)

            # Ensure there is data to apply PCA
            if combined_data.empty:
                st.error("Combined data is empty. Please check the datasets and ensure there are common genes.")
                st.stop()

            # Ensure the data is numerical
            combined_data = combined_data.apply(pd.to_numeric, errors='coerce').dropna()

            # Perform PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(combined_data.T)

            # Explained variance ratio
            explained_variance = pca.explained_variance_ratio_

            # Plotting the PCA results
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', marker='o', label='Samples')
            ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.2f}% variance)')
            ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.2f}% variance)')
            ax.set_title("PCA of Gene Expression Data")
            ax.legend()

            # Show PCA plot in Streamlit
            st.pyplot(fig)

            # K-means Clustering on PCA Results
            kmeans = KMeans(n_clusters=3)  # You can adjust the number of clusters
            kmeans_labels = kmeans.fit_predict(pca_result)

            # Plot clusters
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels, cmap='viridis', label='Clusters')
            ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.2f}% variance)')
            ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.2f}% variance)')
            ax.set_title(f"K-means Clustering of PCA Results (k=3)")
            ax.legend()

            st.pyplot(fig)

            # Differential Expression (t-test example)
            if len(common_genes) > 1:  # Ensure there's more than one gene
                group_labels = kmeans_labels
                de_results = differential_expression_analysis(combined_data, group_labels)
                st.subheader("Differential Expression Results (p-values)")
                st.write(de_results)

            # Gene Ontology Enrichment (Optional)
            enrichment_results = go_enrichment_analysis(common_genes)
            st.subheader("GO Enrichment Results")
            st.write(enrichment_results)

            # Display common genes in a table
            st.subheader("Common Genes")
            st.write(pd.DataFrame(common_genes, columns=["Common Genes"]))

            # Provide an option to download the processed data
            st.subheader("Download Processed Data")
            st.download_button(
                label="💾 Download Combined Processed Data (CSV)",
                data=csv_download(combined_data, "processed_combined_data.csv"),
                file_name="processed_combined_data.csv",
                mime="text/csv"
            )

            # Display combined data as a table (optional)
            st.subheader("Processed Gene Expression Data")
            st.dataframe(combined_data)

        else:
            st.warning("No datasets retrieved.")
