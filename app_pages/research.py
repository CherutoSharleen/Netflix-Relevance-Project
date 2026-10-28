"""Guide private upload/download research workflows without hosting a shared judgment log."""

import csv
import io

import streamlit as st

from failure_review import ablation_deltas
from streamlit_data import ROOT, WorkflowOutput, archive, uploaded_workflow


st.title("Retrieval research", anchor=False)
st.markdown("Prepare candidates, label them yourself with the CLI, then measure and review failures.")
st.warning("No research results are bundled. Missing labels and missing human failure assignments stop the workflow.")
st.caption("Uploads are processed on the app server in temporary files, not committed to GitHub or stored as a shared log. "
           "Download every output: browser sessions and Community Cloud files are not durable storage.")

action = st.selectbox("Workflow step", (
    "Prepare candidate pool", "Evaluate judgments", "Summarize assigned failures", "Freeze final settings",
), key="research_action")
partition = "dev"
uploads: dict[str, bytes] = {}


def clear_output() -> None:
    st.session_state.pop("research_output", None)


if action in {"Prepare candidate pool", "Evaluate judgments"}:
    partition = st.selectbox("Query partition", ("dev", "heldout"), key="research_partition")
    split = st.file_uploader("Frozen query split (split.json)", type="json",
                             key="research_split", on_change=clear_output, max_upload_size=20)
    if split is not None:
        uploads["split"] = split.getvalue()
    if partition == "dev":
        configuration = st.file_uploader("Predeclared experiments JSON", type="json",
                                        key="research_experiments", on_change=clear_output, max_upload_size=2)
        if configuration is not None:
            uploads["experiments"] = configuration.getvalue()
        st.download_button("Download untuned configuration examples", (ROOT / "experiments.json").read_bytes(),
                           "experiments.json", "application/json")
    else:
        st.info("Held-out settings, cutoff, and thresholds are read from your frozen development configuration. "
                "Ablation lists are not accepted here.")
        frozen = st.file_uploader("Frozen final settings (frozen.json)", type="json",
                                 key="research_frozen", on_change=clear_output, max_upload_size=2)
        if frozen is not None:
            uploads["frozen"] = frozen.getvalue()
else:
    saved_report = st.file_uploader("Saved evaluation report (evaluation.json)", type="json",
                                   key="research_report", on_change=clear_output, max_upload_size=20)
    if saved_report is not None:
        uploads["report"] = saved_report.getvalue()

k, relevant_grade, threshold, selected = 10, None, None, ""
if partition == "dev" and action in {"Prepare candidate pool", "Evaluate judgments"}:
    k = st.number_input("Retrieval and evaluation depth k", min_value=1, max_value=100, value=10,
                        key="research_k")
if action == "Prepare candidate pool":
    extra = st.file_uploader("Optional extra ungraded candidates (query_id,show_id CSV)", type="csv",
                            key="research_extra", on_change=clear_output, max_upload_size=20)
    if extra is not None:
        uploads["extra"] = extra.getvalue()
    st.caption("The pool is the union of the chosen runs. No relevance grades are created.")
elif action == "Evaluate judgments":
    for key, label, extension in (
        ("pool", "Initial candidate pool JSON", "json"), ("labels", "Your human judgments CSV", "csv"),
    ):
        upload = st.file_uploader(label, type=extension, key=f"research_{key}",
                                 on_change=clear_output, max_upload_size=20)
        if upload is not None:
            uploads[key] = upload.getvalue()
    if partition == "dev":
        relevant_grade = st.selectbox("Minimum grade counted as relevant", (1, 2, 3), index=None,
                                     key="research_grade")
        threshold = st.number_input("Flag queries with nDCG@k below", min_value=0.0, max_value=1.0,
                                    value=None, step=0.05, key="research_threshold")
    st.caption("All retrieved records must be judged. Recall is relative to the judged list; "
               "nDCG uses linear gain. MRR is limited to the supplied retrieval depth.")
elif action == "Summarize assigned failures":
    upload = st.file_uploader("Your completed failure assignments CSV", type="csv",
                             key="research_assignments", on_change=clear_output, max_upload_size=20)
    if upload is not None:
        uploads["assignments"] = upload.getvalue()
    st.caption("Assign each failing query/experiment a primary cause yourself. Blank causes are not counted as results.")
else:
    selected = st.text_input("Exact development experiment name you selected", key="research_selected")
    st.caption("This freezes your choice and metric protocol. It does not choose the best configuration for you.")

if st.button("Run selected step", type="primary", key="research_run"):
    clear_output()
    try:
        with st.spinner("Validating inputs and running the selected step..."):
            st.session_state["research_output"] = uploaded_workflow(
                action, uploads, partition, int(k), relevant_grade, threshold, selected,
            )
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        st.error(str(error))

output = st.session_state.get("research_output")
if isinstance(output, WorkflowOutput):
    st.subheader("Generated outputs")
    st.caption("These are the last generated outputs, not live results for changed controls. Download and back them up.")
    if output.report is not None:
        report = output.report
        st.markdown(f"**Partition:** {report.partition} · **Depth:** {report.k} · "
                    f"**Relevant grade:** at least {report.relevant_grade}")
        st.dataframe([{"experiment": experiment.name, **experiment.aggregate}
                      for experiment in report.experiments], hide_index=True)
        deltas = ablation_deltas(report)
        if deltas:
            st.subheader("Ablation deltas from the first experiment")
            st.dataframe(deltas, hide_index=True)
        if "failure_counts.csv" in output.files:
            counts = list(csv.DictReader(io.StringIO(output.files["failure_counts.csv"].decode("utf-8"))))
            st.subheader("Human-assigned failure counts")
            st.dataframe(counts, hide_index=True)
            baseline = report.experiments[0].name
            chart_rows = [{"cause": row["cause"], "count": int(row["count"])}
                          for row in counts if row["experiment"] == baseline]
            if chart_rows:
                st.caption(f"Baseline only: {baseline}. Other ablations are not pooled into its failure distribution.")
                st.bar_chart(chart_rows, x="cause", y="count", horizontal=True)
            else:
                st.info("No baseline queries fell below the chosen threshold.")
    st.download_button("Download all outputs (ZIP)", archive(output.files), "research_outputs.zip",
                       "application/zip", key="research_zip")
    for name, content in output.files.items():
        st.download_button(f"Download {name}", content, name, key=f"research_download_{name}")

with st.expander("Protocol and reproducibility"):
    st.code("python label_cli.py --help\npython analysis_cli.py --help", language="powershell")
    st.markdown("Freeze your own query split before tuning. Prepare a union pool before labelling; "
                "use the local CLI for timestamped grades and blind re-labelling. Do not use held-out queries "
                "to choose settings. Download the review sheet and assign all failure causes yourself.")
    for name in ("LABELING.md", "METRICS.md", "ANALYSIS.md"):
        st.download_button(f"Download {name}", (ROOT / name).read_bytes(), name, key=f"protocol_{name}")
