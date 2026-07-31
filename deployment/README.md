# Running Locally

## 1. Backend (FastAPI)

cd deployment/api
docker build -t deepfake-api .
docker run -p 8001:8001 deepfake-api

API will be available at http://localhost:8001
Docs/testing UI: http://localhost:8001/docs

## 2. Frontend (React)

cd deployment/frontend
npm install
npm run dev

Opens at http://localhost:5173 (or next available port — check terminal output)

Note: if the frontend runs on a different port than 5173, update `allow_origins`
in `deployment/api/main.py` to match, or the API will block the request (CORS).

## 3. Streamlit (alternative, all-in-one version)

cd deployment/streamlit
pip install -r requirements.txt
streamlit run app.py

Opens at http://localhost:8501 — runs the model directly, no separate API needed.

---
Model: best_resnet50.pth (christina)
Not yet deployed, running locally only for now.