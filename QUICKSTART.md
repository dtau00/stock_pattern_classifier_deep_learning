# Quick Start Guide - Data Manager UI

## Launch the App

```bash
# From project root directory:
streamlit run src/ui/app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## Testing Workflow

### 1. Download Your First Package (5 minutes)

1. **Navigate** to **⬇️ Configure Download** in the sidebar
2. **Select:**
   - Symbol: `BTCUSDT`
   - Interval: `1h`
   - Start Date: 7 days ago (use date picker)
   - End Date: Today
3. **Review** the estimation:
   - Should show ~168 bars
   - Size: ~8-10 KB
4. **Click** "🚀 Download Data Package"
5. **Wait** ~10-20 seconds for download
6. **Review** the summary:
   - Check total bars
   - Look at validation report
   - Preview first/last 5 rows

### 2. Manage Packages (2 minutes)

1. **Navigate** to **📦 Manage Packages**
2. **View** summary statistics
3. **Click** ⚙️ on your package
4. **Click** "📄 Details" to see full information
5. **Review:**
   - Basic info
   - Validation report
   - Data preview
   - Statistics
6. **Click** "Close Details"

### 3. Validate & Preview (5 minutes)

1. **Navigate** to **✅ Validate & Preview**
2. **Select** your package from dropdown
3. **Review** quality score (should be ~100% for BTCUSDT)
4. **Check** detailed metrics
5. **Explore** the tabs:
   - First 100
   - Last 100
   - Random Sample
   - Statistics
6. **View** the charts:
   - Candlestick chart (zoom, pan, hover)
   - Volume chart
7. **Check** integrity checks (all should pass ✅)
8. **Try** exporting validation report (JSON)

### 4. Download Another Package (Optional)

Test with different configurations:
- Different symbol: `ETHUSDT`
- Different interval: `4h`
- Longer date range: 30 days

### 5. Test Management Features

1. Go to **📦 Manage Packages**
2. Try sorting: "Sort by Total Bars"
3. Try filtering: "Filter by Symbol: BTCUSDT"
4. Delete a package:
   - Click ⚙️ → 🗑️ Delete
   - Confirm deletion
   - Verify it's removed

---

## Expected Results

### Download Page
- ✅ Form validates inputs correctly
- ✅ Estimation updates in real-time
- ✅ Progress bar shows during download
- ✅ Success message appears with summary
- ✅ Validation report shows quality
- ✅ Data preview displays correctly

### Manage Page
- ✅ Summary shows correct counts
- ✅ Packages list displays all downloads
- ✅ Sorting changes order
- ✅ Filtering reduces list
- ✅ Details modal shows complete info
- ✅ Delete removes package and file

### Validate Page
- ✅ Quality score calculated correctly
- ✅ Metrics display actual values
- ✅ Charts render interactively
- ✅ Integrity checks all pass
- ✅ JSON export downloads

---

## Troubleshooting

### App won't start
```bash
# Check Streamlit is installed:
pip list | grep streamlit

# If not installed:
pip install streamlit plotly

# Try again:
streamlit run src/ui/app.py
```

### Import errors
```bash
# Run the test script first:
python test_streamlit_ui.py

# Should show all ✓ checks passing
```

### Download fails
- Check internet connection
- Try a smaller date range (3-7 days)
- Check the error message for details
- Verify symbol is valid on Binance

### Can't see my package
- Check **📦 Manage Packages** page
- Remove filters if active
- Try sorting by "Download Date (Newest)"

---

## Sample Test Data

Good test configurations:

**Small Test (Fast Download)**
- Symbol: BTCUSDT
- Interval: 1h
- Range: Last 7 days
- Expected: ~168 bars, ~8 KB

**Medium Test**
- Symbol: ETHUSDT
- Interval: 1h
- Range: Last 30 days
- Expected: ~720 bars, ~35 KB

**Larger Test**
- Symbol: BTCUSDT
- Interval: 4h
- Range: Last 90 days
- Expected: ~540 bars, ~26 KB

---

## Files Created

After downloading, check these directories:

**Data Packages:**
```
data/packages/
└── BTCUSDT_1h_2025-11-24_2025-12-01.csv
```

**Metadata:**
```
data/metadata/
└── packages.json
```

---

## Next Steps After Testing

Once you've verified the UI works:

1. **Download training data** - Larger datasets for actual training
2. **Proceed to Phase 3** - Feature engineering (returns, volume, volatility)
3. **Implement Phase 4** - Preprocessing pipeline
4. **Build Phase 5** - Data splitting for train/val/test
5. **Create Phase 6** - Advanced visualization tools

---

## Need Help?

- Check [phase2_completion_report.md](docs/phase2_completion_report.md) for detailed documentation
- Review [phase1_completion_report.md](docs/phase1_completion_report.md) for data fetching info
- Check console output for error messages
- Streamlit errors appear in the UI (red boxes)

---

**Happy Testing! 🚀**
