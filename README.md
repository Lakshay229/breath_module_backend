---
title: HCC Biomarker Prediction API
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# HCC Biomarker Prediction API

FastAPI service that predicts liver tumor risk from four sensor readings:
ammonia, hydrogen sulphide, acetone, and methane. Serves Random Forest,
XGBoost, and KNN models trained on the VOC dataset.

## Endpoints

- `GET /` — health check + list of loaded models
- `GET /health` — simple status check
- `POST /predict` — run a prediction

### Example request

```bash
curl -X POST "https://<your-space-url>/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "ammonia": 1.79,
        "hydrogen_sulphide": 0.30,
        "acetone": 1.81,
        "methane": 0.83,
        "model": "rf"
      }'
```

### Example response

```json
{
  "model_used": "rf",
  "prediction": 1,
  "label": "positive",
  "probability": 0.87
}
```

## Setup before deploying

The trained model files are already bundled in the `models/` folder
(`model_random_forest.joblib`, `model_xgboost.joblib`, `model_knn.joblib`,
`scaler.joblib`) — no extra download step needed. Just:

1. Push this folder to a new Hugging Face Space (SDK: Docker).
2. The Space will build automatically and give you a public HTTPS URL.

If you retrain later, drop the new `.joblib` files into `models/` with the
same filenames and redeploy.

## Calling this from Flutter

```dart
final response = await http.post(
  Uri.parse('https://<your-space-url>/predict'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({
    'ammonia': ammoniaValue,
    'hydrogen_sulphide': h2sValue,
    'acetone': acetoneValue,
    'methane': methaneValue,
    'model': 'rf',
  }),
);
final result = jsonDecode(response.body);
```
