# Multi-Modal Climate-Disease Sales Forecasting System

## Architecture
- **Branch 1**: Historical Sales (CNN) - 10 features, 120-day lookback
- **Branch 2**: Climate-Disease (CNN-BiLSTM) - 13 features, disease risk modeling
- **Branch 3**: Temporal Context (BiLSTM) - 8 features, holidays/month-end/promotions
- **Fusion**: Cross-Attention with dynamic branch weighting
- **Output**: Daily sales forecast + confidence bounds + explainability

## Quick Start

### 1. Install Python Dependencies
```bash
cd python
pip install -r requirements.txt
```

### 2. Train the Model
```bash
cd python
python train.py
```

This will:
- Load weather data from `data/weather/open-meteo-6.9355N79.8487E.csv`
- Build training sequences (120-day lookback, 30-day horizon)
- Train the multi-modal CNN-BiLSTM-Attention model
- Export to ONNX at `models/multimodal_forecast.onnx`

### Refresh Training Snapshot
Before retraining against a newer ERP database, refresh the repo-local sales and
historical weather snapshot:

```bash
python python/refresh_training_snapshot.py
```

This refreshes `dbo.ProRptDailysalessummary` branch sales from `2025-01-01`,
writes a deterministic parquet cache under `data/cache/`, fetches Open-Meteo
historical weather through the latest SP sales date, and writes
`data/weather/open-meteo-6.9355N79.8487E.csv`, and writes
`data/cache/training_snapshot_manifest.json`.

For a scheduled refresh, run the same command every other day after ERP posting,
for example at 06:00 Asia/Colombo. If the latest SP date has not advanced since
the previous manifest, the script exits without rewriting inputs. If Open-Meteo
historical weather is behind the DB cutoff, the script fails unless clipping is
explicitly allowed:

```bash
python python/refresh_training_snapshot.py --allow-cutoff-to-weather
```

### 3. Run the .NET API
```bash
cd src/PharmaAI.Forecasting.Api
dotnet run
```

API available at: http://localhost:5000

### 4. Generate Forecast
```bash
curl -X POST "http://localhost:5000/api/forecast?seriesId=company&location=LK-CO&horizonDays=30"
```

## Project Structure
```
├── data/
│   ├── weather/          # Open-Meteo CSV data
│   ├── atc/              # WHO ATC classification mapping
│   └── config/           # Configuration files
├── python/
│   ├── src/
│   │   ├── data_pipeline.py      # Data loading and sequence building
│   │   ├── model.py              # Multi-modal CNN-BiLSTM-Attention model
│   │   ├── trainer.py            # Training loop with early stopping
│   │   └── export_onnx.py        # ONNX export for .NET inference
│   ├── train.py                  # Main training script
│   └── requirements.txt
├── src/
│   ├── PharmaAI.Forecasting.Api/     # ASP.NET Core Web API
│   │   ├── Controllers/
│   │   │   └── ForecastController.cs
│   │   ├── Services/
│   │   │   └── OnnxForecastService.cs
│   │   ├── Program.cs
│   │   └── appsettings.json
│   └── PharmaAI.Forecasting.Core/    # Shared models and interfaces
│       ├── Models/
│       │   └── ForecastResponse.cs
│       └── Interfaces/
│           └── IForecastService.cs
├── models/                   # Trained model files (.pt, .onnx)
└── PharmaAI.Forecasting.sln
```

## Model Architecture
- **Input**: 3 branches (sales: 120x10, climate: 120x13, temporal: 120x8)
- **Branch 1**: Conv1D(64,7) → Conv1D(128,14) → Conv1D(128,30) → AdaptiveAvgPool
- **Branch 2**: Conv1D(64,7) → Conv1D(128,14) → Conv1D(128,30) → BiLSTM(128) → BiLSTM(128) → BiLSTM(64)
- **Branch 3**: BiLSTM(64, bidirectional)
- **Fusion**: Concat → Attention(3) → Weighted sum → Dense(256) → Dense(128) → Dense(1)
- **Total parameters**: ~500K

## Disease-Climate Mapping
| Disease Category | ATC Codes | Lag Window | Climate Trigger |
|-----------------|-----------|------------|-----------------|
| Respiratory | R01,R03,R05,R06,J01 | 5-10 days | Temp < 24°C |
| Vector-borne | P01,J05,B05 | 30-70 days | Rainfall > 40mm |
| Water-borne | A07,J01,A03 | 30-60 days | Monsoon onset |
| Dermatological | D01,D06,D08 | 7-14 days | Humidity > 80% + Rain |
| Heat-related | B05,A07,A03 | 0-5 days | Temp > 34°C |

## Configuration
See `src/PharmaAI.Forecasting.Api/appsettings.json` for:
- Climate thresholds
- Disease lag windows
- Region configuration (multi-city ready)
- Weather API settings
