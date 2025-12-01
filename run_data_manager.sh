#!/bin/bash
# Launch Data Manager Streamlit App
cd "$(dirname "$0")"
streamlit run src/ui/app.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false
