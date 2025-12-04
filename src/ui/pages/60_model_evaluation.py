"""
Model Evaluation & Validation Dashboard

Comprehensive evaluation of trained models with all metrics from Design Document Section 5:
- Clustering quality metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Confidence calibration quality (R²)
- Statistical independence tests (Kruskal-Wallis, Chi-Square)
- Stability testing (ARI)

Reference: Design Document Section 5 & 6
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import (
    ClusteringMetrics,
    StatisticalTests,
    EvaluationReport
)


def main():
    st.title("📊 Model Evaluation & Validation")
    st.markdown("Comprehensive evaluation with Design Document metrics")

    # Check if model is trained
    if not st.session_state.get('training_complete', False):
        st.warning("⚠️ No trained model found. Please train a model first (Page: Model Training)")
        return

    # Tabs for different evaluations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Clustering Quality",
        "🎯 Confidence Calibration",
        "🔬 Statistical Tests",
        "📋 Full Report"
    ])

    with tab1:
        show_clustering_quality()

    with tab2:
        show_confidence_calibration()

    with tab3:
        show_statistical_tests()

    with tab4:
        show_full_report()


def show_clustering_quality():
    """Display clustering quality metrics."""
    st.header("Clustering Quality Metrics")
    st.markdown("Metrics from Design Document Section 5")

    # Get validation data
    if 'val_loader' not in st.session_state:
        st.warning("Validation data not available")
        return

    model = st.session_state['model']
    val_loader = st.session_state['val_loader']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if st.button("🔍 Compute Clustering Metrics", type="primary"):
        with st.spinner("Computing metrics..."):
            # Collect validation latent vectors
            z_list = []
            cluster_ids_list = []

            model.eval()
            with torch.no_grad():
                for (x,) in val_loader:
                    x = x.to(device)
                    z = model.encoder(x)
                    z_norm = F.normalize(z, p=2, dim=1)
                    cluster_ids = model.get_cluster_assignment(z_norm)

                    z_list.append(z_norm.cpu())
                    cluster_ids_list.append(cluster_ids.cpu())

            z_val = torch.cat(z_list, dim=0).numpy()
            cluster_ids_val = torch.cat(cluster_ids_list, dim=0).numpy()

            # Compute metrics
            metrics_calc = ClusteringMetrics()
            metrics = metrics_calc.compute_all_metrics(z_val, cluster_ids_val)

            # Check thresholds
            threshold_results = metrics_calc.check_metric_thresholds(metrics)

            # Store results
            st.session_state['clustering_metrics'] = metrics
            st.session_state['threshold_results'] = threshold_results

    # Display results if available
    if 'clustering_metrics' in st.session_state:
        metrics = st.session_state['clustering_metrics']
        threshold_results = st.session_state['threshold_results']

        st.divider()

        # Metrics cards
        col1, col2, col3 = st.columns(3)

        with col1:
            status = "✅" if threshold_results['silhouette_score'][0] else "❌"
            st.metric(
                "Silhouette Score",
                f"{metrics['silhouette_score']:.3f}",
                delta="Good" if threshold_results['silhouette_score'][0] else "Poor",
                help="Range: [-1, 1], Threshold: ≥ 0.4"
            )
            st.caption(f"{status} {threshold_results['silhouette_score'][1]}")

        with col2:
            status = "✅" if threshold_results['davies_bouldin_index'][0] else "❌"
            st.metric(
                "Davies-Bouldin Index",
                f"{metrics['davies_bouldin_index']:.3f}",
                delta="Good" if threshold_results['davies_bouldin_index'][0] else "Poor",
                help="Lower is better, Threshold: ≤ 1.5"
            )
            st.caption(f"{status} {threshold_results['davies_bouldin_index'][1]}")

        with col3:
            status = "✅" if threshold_results['calinski_harabasz_score'][0] else "❌"
            st.metric(
                "Calinski-Harabasz Score",
                f"{metrics['calinski_harabasz_score']:.1f}",
                delta="Good" if threshold_results['calinski_harabasz_score'][0] else "Poor",
                help="Higher is better, Threshold: ≥ 100"
            )
            st.caption(f"{status} {threshold_results['calinski_harabasz_score'][1]}")

        # Overall status
        all_passed = all(passed for passed, _ in threshold_results.values())
        if all_passed:
            st.success("✅ All clustering quality metrics passed!")
        else:
            st.error("❌ Some clustering quality metrics failed. Model may need improvement.")


def show_confidence_calibration():
    """Display confidence calibration results."""
    st.header("Confidence Calibration Quality")
    st.markdown("Calibration results from training")

    calibration = st.session_state.get('calibration')

    if calibration is None:
        st.warning("Confidence calibration not available. Please retrain the model.")
        return

    # Calibration summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best Gamma (γ*)",
            f"{calibration['best_gamma']:.1f}",
            help="Calibrated gamma parameter for sigmoid"
        )

    with col2:
        st.metric(
            "R² Score",
            f"{calibration['best_r2']:.3f}",
            delta="Good" if calibration['best_r2'] >= 0.7 else "Poor",
            help="Correlation with Silhouette Score, Threshold: ≥ 0.7"
        )

    with col3:
        st.metric(
            "Regression Slope",
            f"{calibration['best_slope']:.2f}",
            delta="Positive" if calibration['best_slope'] > 0 else "Negative",
            help="Should be positive (higher confidence = higher quality)"
        )

    # Status
    st.divider()
    if calibration['passed']:
        st.success(
            f"✅ Confidence calibration PASSED! "
            f"(R² = {calibration['best_r2']:.3f} ≥ 0.7)"
        )
        st.info(
            f"Confidence scores are calibrated and reliable. "
            f"Use gamma = {calibration['best_gamma']:.1f} for inference."
        )
    else:
        st.error(
            f"❌ Confidence calibration FAILED! "
            f"(R² = {calibration['best_r2']:.3f} < 0.7)"
        )
        st.warning(
            "Model training has FAILED - confidence scores are not reliable. "
            "Recommend retraining with different hyperparameters."
        )

    # Detailed results
    with st.expander("📊 Detailed Calibration Results"):
        st.subheader("Grid Search Results")

        all_results = calibration.get('all_results', [])
        if all_results:
            results_df = pd.DataFrame([
                {
                    'Gamma': r['gamma'],
                    'R²': r['r2'],
                    'Slope': r['slope'],
                    'Intercept': r['intercept']
                }
                for r in all_results
            ])
            st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.subheader("Validation Set Statistics")
        st.write(f"- Number of samples: {calibration['n_samples']:,}")
        st.write(f"- Mean Silhouette Score: {calibration['silhouette_mean']:.3f}")
        st.write(f"- Std Silhouette Score: {calibration['silhouette_std']:.3f}")


def show_statistical_tests():
    """Display statistical independence tests."""
    st.header("Statistical Independence Tests")
    st.markdown("Tests from Design Document Section 5")

    st.info(
        "⚠️ Statistical tests require additional data:\n"
        "- Volatility Independence: NATR values\n"
        "- Temporal Stability: Timestamps\n\n"
        "These tests validate that clustering is based on pattern shape, "
        "not magnitude or temporal artifacts."
    )

    # Placeholder for when data is available
    st.subheader("Volatility Independence Test")
    st.markdown("**Kruskal-Wallis H-test**")
    st.write("Tests if clustering is independent of volatility (NATR).")
    st.caption("Status: Not implemented (requires NATR values from preprocessing)")

    st.divider()

    st.subheader("Temporal Stability Test")
    st.markdown("**Chi-Square Test of Independence**")
    st.write("Tests if clustering is independent of time periods.")
    st.caption("Status: Not implemented (requires timestamps from preprocessing)")

    st.divider()

    st.info(
        "💡 **To enable statistical tests:**\n"
        "1. Include NATR values and timestamps in preprocessed data\n"
        "2. Pass them to the evaluation pipeline\n"
        "3. Tests will run automatically and show pass/fail status"
    )


def show_full_report():
    """Generate and display full evaluation report."""
    st.header("Full Evaluation Report")
    st.markdown("Comprehensive evaluation with all Design Document metrics")

    if st.button("📋 Generate Full Report", type="primary"):
        with st.spinner("Generating comprehensive evaluation..."):
            try:
                # Get model and data
                model = st.session_state['model']
                val_loader = st.session_state['val_loader']
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Collect validation data
                z_list = []
                cluster_ids_list = []

                model.eval()
                with torch.no_grad():
                    for (x,) in val_loader:
                        x = x.to(device)
                        z = model.encoder(x)
                        z_norm = F.normalize(z, p=2, dim=1)
                        cluster_ids = model.get_cluster_assignment(z_norm)

                        z_list.append(z_norm.cpu())
                        cluster_ids_list.append(cluster_ids.cpu())

                z_val = torch.cat(z_list, dim=0).numpy()
                cluster_ids_val = torch.cat(cluster_ids_list, dim=0).numpy()
                centroids = F.normalize(model.centroids, p=2, dim=1).cpu().numpy()

                # Get calibration results
                calibration = st.session_state.get('calibration')

                # Generate report
                evaluator = EvaluationReport()
                report = evaluator.evaluate_complete(
                    z_normalized_val=z_val,
                    cluster_ids_val=cluster_ids_val,
                    centroids_normalized=centroids,
                    calibration_results=calibration
                )

                # Store report
                st.session_state['evaluation_report'] = report

                # Save to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = Path(f"data/reports/evaluation_{timestamp}.json")
                evaluator.save_report(report, str(report_path))

            except Exception as e:
                st.error(f"Report generation failed: {e}")
                st.exception(e)
                return

    # Display report if available
    if 'evaluation_report' in st.session_state:
        report = st.session_state['evaluation_report']

        st.divider()

        # Overall status
        status = report['overall_status']
        if status == 'PASS':
            st.success(f"✅ Overall Status: {status}")
            st.balloons()
            st.info("🎉 Model is ready for production use!")
        else:
            st.error(f"❌ Overall Status: {status}")

        # Show failures
        if report['failures']:
            st.subheader("❌ Failures")
            for failure in report['failures']:
                st.error(f"- {failure}")

        # Validation set metrics
        st.subheader("📊 Validation Set Metrics")

        val_metrics = report['validation_set']['metrics']
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Silhouette", f"{val_metrics['silhouette_score']:.3f}")
        col2.metric("Davies-Bouldin", f"{val_metrics['davies_bouldin_index']:.3f}")
        col3.metric("Calinski-Harabasz", f"{val_metrics['calinski_harabasz_score']:.1f}")
        col4.metric("Samples", f"{val_metrics['n_samples']:,}")

        # Confidence calibration
        if 'confidence_calibration' in report:
            st.subheader("🎯 Confidence Calibration")

            calib = report['confidence_calibration']
            col1, col2, col3 = st.columns(3)

            col1.metric("Best Gamma", f"{calib['best_gamma']:.1f}")
            col2.metric("R²", f"{calib['best_r2']:.3f}")
            col3.metric("Status", "PASS" if calib['passed'] else "FAIL")

        # Download report
        st.divider()
        if st.button("💾 Download Report JSON"):
            import json
            report_json = json.dumps(report, indent=2)

            st.download_button(
                label="Download evaluation_report.json",
                data=report_json,
                file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Model Evaluation",
        page_icon="📊",
        layout="wide"
    )
    main()
