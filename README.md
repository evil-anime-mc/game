# 🎮 GameCheck — Should I Buy This Game?

A data-driven game analysis tool that pulls Steam reviews, runs ML sentiment analysis, and delivers an AI-powered verdict on whether a game is worth buying.

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | Python · FastAPI · VADER Sentiment |
| AI Verdict | Claude API (claude-sonnet) |
| Data | Steam Store API (free, no key needed) |
| Frontend | Vanilla HTML/CSS/JS · Chart.js |
| Deploy | Railway (backend) · Netlify (frontend) |

## Features

- 🔍 **Steam Game Search** — Find any game instantly
- 🧠 **ML Sentiment Analysis** — VADER scores 100+ reviews
- 📊 **8-Month Trend Chart** — See if reviews are improving or declining
- 🤖 **Claude AI Verdict** — BUY NOW / WAIT FOR SALE / SKIP
- 💬 **Top Review Quotes** — Best and worst things players say
- 🎮 **Gaming Aesthetic UI** — Dark, scanline, glowing dashboard

---

## Local Development

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
uvicorn main:app --reload
# Runs on http://localhost:8000
```

### 2. Frontend

Just open `frontend/index.html` in your browser — no build step needed.

If your backend is on a different URL, change the `API` constant at the top of `index.html`:
```js
const API = 'http://localhost:8000';
```

---

## Deploy to Production

### Backend → Railway

1. Go to [railway.app](https://railway.app) and create a new project
2. Connect your GitHub repo
3. Set environment variable: `ANTHROPIC_API_KEY=your_key`
4. Railway auto-detects the Dockerfile and deploys
5. Copy your Railway URL (e.g. `https://gamechecker-api.up.railway.app`)

### Frontend → Netlify

1. Go to [netlify.com](https://netlify.com)
2. Drag and drop the `frontend/` folder
3. Open `index.html`, update the `API` constant to your Railway URL:
```js
const API = 'https://your-app.up.railway.app';
```
4. Redeploy

---

## CV Description

> Built a full-stack game analysis platform that performs NLP sentiment analysis on Steam reviews using VADER, visualises 8-month sentiment trends, and integrates the Claude AI API to generate data-driven buy/skip verdicts. Stack: Python, FastAPI, vanilla JS, Chart.js.

## Algorithms Used (for your CV)

- **VADER Sentiment Analysis** — Valence Aware Dictionary and Sentiment Reasoner
- **Compound Score Normalisation** — Maps [-1, 1] → [0, 100] scale
- **Temporal Aggregation** — Monthly sentiment bucketing for trend analysis
- **LLM Prompting** — Structured data → Claude → JSON verdict

---

*Data sourced from Steam Store API. No Steam API key required.*
