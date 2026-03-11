import os
import re
import json
from collections import defaultdict
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI(title="GameChecker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = SentimentIntensityAnalyzer()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


@app.get("/")
def health():
    return {"status": "ok", "service": "GameChecker API"}


@app.get("/search")
def search_games(game: str = Query(..., min_length=1)):
    url = "https://store.steampowered.com/api/storesearch/"
    r = requests.get(url, params={"term": game, "l": "en", "cc": "US"}, timeout=10)
    if r.status_code != 200:
        raise HTTPException(502, "Steam search failed")
    items = r.json().get("items", [])[:6]
    results = []
    for item in items:
        price_info = item.get("price", {})
        if isinstance(price_info, dict):
            final = price_info.get("final", 0)
            price_str = "${:.2f}".format(final / 100) if final else "Free"
        else:
            price_str = "N/A"
        results.append({
            "appid": item["id"],
            "name": item["name"],
            "price": price_str,
            "tiny_image": item.get("tiny_image", ""),
        })
    return results


@app.get("/game")
def game_details(appid: int):
    url = "https://store.steampowered.com/api/appdetails"
    r = requests.get(url, params={"appids": appid, "cc": "us", "l": "en"}, timeout=10)
    data = r.json().get(str(appid), {})
    if not data.get("success"):
        raise HTTPException(404, "Game not found")
    d = data["data"]
    price_data = d.get("price_overview", {})
    return {
        "name": d.get("name"),
        "description": d.get("short_description", ""),
        "header_image": d.get("header_image", ""),
        "price": price_data.get("final_formatted", "Free"),
        "discount": price_data.get("discount_percent", 0),
        "genres": [g["description"] for g in d.get("genres", [])],
        "developers": d.get("developers", []),
        "release_date": d.get("release_date", {}).get("date", ""),
        "platforms": d.get("platforms", {}),
        "metacritic": d.get("metacritic", {}).get("score"),
        "recommendations": d.get("recommendations", {}).get("total", 0),
    }


@app.get("/reviews")
def game_reviews(appid: int):
    url = "https://store.steampowered.com/appreviews/{}".format(appid)
    r = requests.get(url, params={
        "json": 1, "filter": "recent", "language": "english",
        "num_per_page": 100, "purchase_type": "all"
    }, timeout=15)
    raw = r.json()
    reviews_data = raw.get("reviews", [])

    if not reviews_data:
        return {
            "total": 0, "score": 50,
            "positive_pct": 50, "negative_pct": 50, "neutral_pct": 0,
            "top_positive": [], "top_negative": [],
            "monthly_trend": [], "summary": raw.get("query_summary", {})
        }

    scores, positives, negatives = [], [], []
    monthly = defaultdict(list)

    for rev in reviews_data:
        text = rev.get("review", "").strip()
        if not text:
            continue
        vs = analyzer.polarity_scores(text)
        compound = vs["compound"]
        scores.append(compound)

        ts = rev.get("timestamp_created", 0)
        if ts:
            month = datetime.utcfromtimestamp(ts).strftime("%Y-%m")
            monthly[month].append(compound)

        entry = {"text": text[:220], "score": round((compound + 1) * 50)}
        if compound >= 0.3:
            positives.append(entry)
        elif compound <= -0.2:
            negatives.append(entry)

    avg = sum(scores) / len(scores) if scores else 0
    overall = round((avg + 1) * 50)
    pos_pct = round(len([s for s in scores if s >= 0.05]) / len(scores) * 100)
    neg_pct = round(len([s for s in scores if s <= -0.05]) / len(scores) * 100)
    neu_pct = 100 - pos_pct - neg_pct

    trend = sorted([
        {"month": m, "score": round((sum(v) / len(v) + 1) * 50)}
        for m, v in monthly.items()
    ], key=lambda x: x["month"])[-8:]

    positives.sort(key=lambda x: x["score"], reverse=True)
    negatives.sort(key=lambda x: x["score"])

    return {
        "total": len(scores),
        "score": overall,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "neutral_pct": neu_pct,
        "top_positive": positives[:3],
        "top_negative": negatives[:3],
        "monthly_trend": trend,
        "summary": raw.get("query_summary", {}),
    }


@app.get("/verdict")
def get_verdict(
    name: str,
    price: str,
    score: int,
    positive_pct: int,
    negative_pct: int,
    discount: int,
    recommendations: int,
    metacritic: str = "N/A",
    genres: str = "",
):
    if not GROQ_API_KEY:
        return {"label": "UNKNOWN", "reason": "GROQ_API_KEY not set.", "tip": ""}

    try:
        prompt = (
            "You are a game analyst. Based on this Steam data, give a verdict.\n"
            "Game: " + name + "\n"
            "Price: " + price + " | Discount: " + str(discount) + "%\n"
            "Sentiment: " + str(score) + "/100 | Positive: " + str(positive_pct) + "% | Negative: " + str(negative_pct) + "%\n"
            "Recommendations: " + str(recommendations) + " | Metacritic: " + str(metacritic) + "\n"
            "Genres: " + genres + "\n\n"
            "Reply ONLY with this JSON:\n"
            "{\"label\": \"BUY NOW\", \"reason\": \"one sentence reason\", \"tip\": \"one tip\"}\n"
            "label must be exactly: BUY NOW or WAIT FOR SALE or SKIP"
        )

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": "Bearer " + GROQ_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "model": "llama3-70b-8192",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15,
        )

        resp_json = response.json()
        if "choices" not in resp_json:
            raise HTTPException(500, "Groq response: " + str(resp_json))
        raw_text = resp_json["choices"][0]["message"]["content"]
        match = re.search(r"\{[^{}]+\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"label": "UNKNOWN", "reason": raw_text, "tip": ""}

    except Exception as e:
        raise HTTPException(500, "Groq error: " + str(e))
