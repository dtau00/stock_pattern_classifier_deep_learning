"""
Feature Visualization Page (formerly TA Verification)

Interactive TradingView-style charts showing:
- Raw OHLCV data
- Engineered features (returns, volume/liquidity, volatility)
- Feature correlation analysis
- Real-time visualization of selected data packages
"""
import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.features import engineer_all_features, compute_feature_correlation
    from src.visualization import (
        create_tradingview_chart,
        create_features_chart,
        create_comparison_chart
    )
    FEATURES_AVAILABLE = True
except ImportError:
    # Fallback to our basic visualization if features module not available
    from src.visualization.data_viz import plot_ta_verification
    FEATURES_AVAILABLE = False

# Import shared component
from src.ui.components import load_package_with_date_range


def show():
    """Feature Visualization - Interactive TradingView-style charts"""
    st.title("Feature Visualization")
    st.markdown("---")

    # Use reusable component for loading package with date range
    load_package_with_date_range(
        packages_dir=str(project_root / "data" / "packages"),
        session_key='feature_viz_data',
        key_prefix='feature_viz',
        file_extension='.csv',
        package_type="OHLCV"
    )

    # Get data from session state
    df = st.session_state.get('feature_viz_data', None)

    # If we have data, process and display
    if df is not None:
        st.markdown("---")
        st.markdown("## 🔧 Feature Engineering")

        # Verify required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.info("Required columns: open, high, low, close, volume")
        else:
            # Feature engineering parameters
            with st.expander("⚙️ Feature Engineering Parameters"):
                col1, col2 = st.columns(2)
                with col1:
                    obv_ema_period = st.slider(
                        "OBV EMA Period",
                        min_value=5,
                        max_value=50,
                        value=20,
                        help="Shorter = more responsive, Longer = more smoothed"
                    )
                with col2:
                    natr_period = st.slider(
                        "NATR Period",
                        min_value=7,
                        max_value=30,
                        value=14,
                        help="Period for volatility calculation"
                    )

            # Apply feature engineering
            with st.spinner("🔄 Applying feature engineering..."):
                try:
                    df_features = engineer_all_features(
                        df.copy(),
                        obv_ema_period=obv_ema_period,
                        natr_period=natr_period
                    )

                    st.success("✓ Feature engineering complete!")

                    # Feature statistics
                    with st.expander("📊 Feature Statistics"):
                        feature_stats = df_features[['returns', 'volume_liquidity', 'volatility_risk']].describe()
                        st.dataframe(feature_stats, use_container_width=True)

                    # Feature correlation
                    st.markdown("### 🔗 Feature Correlation Analysis")

                    corr_matrix, report = compute_feature_correlation(df_features)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Max Correlation",
                            f"{report['max_correlation']:.3f}",
                            delta="Safe" if report['safe_to_train'] else "Warning",
                            delta_color="normal" if report['safe_to_train'] else "inverse"
                        )
                    with col2:
                        st.metric(
                            "Safe to Train",
                            "✓ Yes" if report['safe_to_train'] else "✗ No",
                        )
                    with col3:
                        coverage = df_features['returns'].notna().sum() / len(df_features) * 100
                        st.metric("Data Coverage", f"{coverage:.1f}%")

                    # Show correlation matrix
                    st.dataframe(
                        corr_matrix.style.background_gradient(cmap='RdYlGn_r', vmin=-1, vmax=1),
                        use_container_width=True
                    )

                    if report['warnings']:
                        st.warning("⚠️ " + "\n".join(report['warnings']))
                    if report['critical']:
                        st.error("❌ " + "\n".join(report['critical']))

                    # Chart selection
                    st.markdown("---")
                    st.markdown("## 📈 Interactive Charts")

                    chart_type = st.selectbox(
                        "Select chart type:",
                        [
                            "All Features (Recommended)",
                            "Raw OHLCV Only",
                            "Price vs Returns",
                            "Price vs Volume/Liquidity",
                            "Price vs Volatility/Risk"
                        ]
                    )

                    # Chart height control
                    chart_height = st.slider(
                        "Chart Height (px)",
                        min_value=600,
                        max_value=1400,
                        value=1000,
                        step=100
                    )

                    # Generate selected chart
                    with st.spinner("📊 Generating chart..."):
                        if chart_type == "All Features (Recommended)":
                            fig = create_features_chart(
                                df_features,
                                title="Feature Engineering - All Channels",
                                height=chart_height
                            )
                        elif chart_type == "Raw OHLCV Only":
                            fig = create_tradingview_chart(
                                df_features,
                                title="OHLCV Candlestick Chart",
                                show_volume=True,
                                height=chart_height
                            )
                        elif chart_type == "Price vs Returns":
                            fig = create_comparison_chart(
                                df_features,
                                feature='returns',
                                height=chart_height
                            )
                        elif chart_type == "Price vs Volume/Liquidity":
                            fig = create_comparison_chart(
                                df_features,
                                feature='volume_liquidity',
                                height=chart_height
                            )
                        elif chart_type == "Price vs Volatility/Risk":
                            fig = create_comparison_chart(
                                df_features,
                                feature='volatility_risk',
                                height=chart_height
                            )

                        # Display chart
                        st.plotly_chart(fig, use_container_width=True)

                    # Export options
                    st.markdown("---")
                    st.markdown("## 💾 Export Options")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # Export features as CSV
                        csv = df_features.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Features CSV",
                            data=csv,
                            file_name="features_engineered.csv",
                            mime="text/csv",
                            help="Download DataFrame with engineered features"
                        )

                    with col2:
                        # Export chart as HTML
                        chart_html = fig.to_html()
                        st.download_button(
                            label="📊 Download Chart HTML",
                            data=chart_html,
                            file_name="chart_interactive.html",
                            mime="text/html",
                            help="Download interactive chart as standalone HTML"
                        )

                    with col3:
                        # Export correlation report
                        report_text = f"""Feature Correlation Report

Max Correlation: {report['max_correlation']:.3f}
Safe to Train: {report['safe_to_train']}

Correlation Matrix:
{corr_matrix.to_string()}

Warnings: {len(report['warnings'])}
Critical Issues: {len(report['critical'])}
"""
                        st.download_button(
                            label="📋 Download Report",
                            data=report_text,
                            file_name="correlation_report.txt",
                            mime="text/plain",
                            help="Download correlation analysis report"
                        )

                    # Data preview
                    with st.expander("🔍 Data Preview (First/Last 10 rows)"):
                        st.markdown("**First 10 rows:**")
                        st.dataframe(
                            df_features[['timestamp', 'close', 'returns', 'volume_liquidity', 'volatility_risk']].head(10),
                            use_container_width=True
                        )
                        st.markdown("**Last 10 rows:**")
                        st.dataframe(
                            df_features[['timestamp', 'close', 'returns', 'volume_liquidity', 'volatility_risk']].tail(10),
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"❌ Error during feature engineering: {e}")
                    st.exception(e)

    else:
        # Help section when no data loaded
        st.markdown("---")

if __name__ == "__main__":
    show()
