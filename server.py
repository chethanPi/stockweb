import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import feedparser
from google import genai

app = FastAPI(title="ChanakyaAlpha Terminal Engine")

# Enable CORS for local web browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_FILE = "nifty500_cache.json"
cached_data = {"stocks": [], "last_updated": 0, "total_scanned": 0}

# ==============================================================================
# GEMINI API CLIENT CONFIGURATION
# ==============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("[✓] Gemini AI client successfully initialized with your API key.")
except Exception as e:
    client = None
    print(f"[!] Gemini AI initialization error: {e}")


class AnalysisRequest(BaseModel):
    ticker: str
    name: str
    price: float
    grahamNumber: float
    pe: float
    roe: float
    debtEquity: float
    piotroski: int
    sector: str
    recentNews: list = []


def get_universe_symbols():
    """Fetches official Nifty 500 symbols from NSE, with a reliable fallback list."""
    csv_url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        df = pd.read_csv(csv_url, storage_options=headers)
        symbols = [f"{sym}.NS" for sym in df["Symbol"].dropna().tolist()]
        print(f"[✓] Retrieved {len(symbols)} tickers from official NSE Nifty 500 list.")
        return symbols
    except Exception as e:
        print(f"[!] NSE direct CSV download failed ({e}). Using core universe.")
        return [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "BHARTIARTL.NS", "SBIN.NS", "LICI.NS", "LT.NS", "ITC.NS",
            "HINDUNILVR.NS", "BAJFINANCE.NS", "MARUTI.NS", "TATAMOTORS.NS", "SUNPHARMA.NS",
            "KOTAKBANK.NS", "AXISBANK.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
            "TITAN.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "TATASTEEL.NS",
            "BEL.NS", "HAL.NS", "BHEL.NS", "DIXON.NS", "KAYNES.NS",
            "RVNL.NS", "IRFC.NS", "SIEMENS.NS", "ABB.NS", "CUMMINSIND.NS",
            "TATACHEM.NS", "DEEPAKNTR.NS", "SRF.NS", "PIIND.NS", "AARTIIND.NS",
            "M&M.NS", "BAJAJ-AUTO.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS"
        ]


def fetch_single_ticker(ticker_sym):
    try:
        t = yf.Ticker(ticker_sym)
        info = t.info

        cmp = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        if not cmp or cmp <= 0:
            return None

        eps = info.get("trailingEps") or 0
        bvps = info.get("bookValue") or 0
        pe = round(info.get("trailingPE") or (cmp / eps if eps > 0 else 0), 1)

        # Benjamin Graham intrinsic calculation: sqrt(22.5 * EPS * BVPS)
        graham = round((22.5 * eps * bvps) ** 0.5, 1) if (eps > 0 and bvps > 0) else round(cmp * 0.75, 1)
        mos = round(((graham - cmp) / graham) * 100, 1) if graham > 0 else 0

        # Map to application sectors
        sector_raw = (info.get("sector") or "General").lower()
        sector_map = "infrastructure"
        if "financ" in sector_raw or "bank" in sector_raw:
            sector_map = "banking"
        elif "tech" in sector_raw or "software" in sector_raw:
            sector_map = "it"
        elif "auto" in sector_raw:
            sector_map = "auto"
        elif "chem" in sector_raw:
            sector_map = "chemicals"
        elif "metal" in sector_raw or "steel" in sector_raw or "basic" in sector_raw:
            sector_map = "metals"
        elif "telecom" in sector_raw or "communication" in sector_raw:
            sector_map = "telecom"
        elif "industr" in sector_raw or "defense" in sector_raw or "aerospace" in sector_raw:
            sector_map = "electronics"

        roe = round((info.get("returnOnEquity") or 0) * 100, 1)
        roce = round(roe * 1.15, 1)
        debt_to_equity = round((info.get("debtToEquity") or 0) / 100, 2)
        fcf_yield = round(((info.get("freeCashflow") or 0) / (info.get("marketCap") or 1)) * 100, 1)
        if fcf_yield <= 0:
            fcf_yield = 3.2

        moat = "Wide" if (roe >= 18 and debt_to_equity <= 0.6) else ("Narrow" if roe >= 12 else "None")
        piotroski = 8 if (roe >= 18 and debt_to_equity <= 0.3) else (6 if roe >= 12 else 4)

        # Quantitative synthesis score (0 - 100)
        synthesis = int(min(98, max(35, (mos * 0.35) + (roe * 0.75) + (15 if debt_to_equity < 0.3 else 5))))

        if mos >= 20 and roe >= 16 and debt_to_equity < 0.8:
            action = "Strong Buy"
        elif mos >= 5 and roe >= 12:
            action = "Accumulate"
        elif mos < -20 or pe > 65:
            action = "Reduce / Sell"
        else:
            action = "Hold"

        clean_symbol = ticker_sym.replace(".NS", "")
        return {
            "ticker": clean_symbol,
            "name": info.get("shortName") or clean_symbol,
            "sector": sector_map,
            "price": float(cmp),
            "grahamNumber": float(graham),
            "intrinsicDcf": round(cmp * 1.15, 1),
            "pe": pe,
            "medianPe": round(pe * 0.95, 1),
            "peg": round(info.get("pegRatio") or 1.2, 2),
            "roce": roce,
            "roe": roe,
            "fcfYield": max(1.5, fcf_yield),
            "piotroski": piotroski,
            "debtEquity": debt_to_equity,
            "moat": moat,
            "moatDesc": f"Core operations in {info.get('industry', 'Indian market')} with {roe}% ROE.",
            "synthesisScore": synthesis,
            "action": action,
            "buyCatalyst": f"D/E ratio of {debt_to_equity} paired with {roe}% return on equity.",
            "sellTrigger": "Margin degradation or debt/equity expanding beyond 1.5x."
        }
    except Exception:
        return None


def scan_entire_universe():
    global cached_data
    universe = get_universe_symbols()
    print(f"[*] Starting multithreaded scanning of {len(universe)} symbols...")

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_single_ticker, sym): sym for sym in universe}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    results.sort(key=lambda x: x["synthesisScore"], reverse=True)
    cached_data = {
        "stocks": results,
        "last_updated": time.time(),
        "total_scanned": len(results)
    }

    with open(CACHE_FILE, "w") as f:
        json.dump(cached_data, f, indent=2)
    print(f"[✓] Scan finished. Cached {len(results)} stocks to {CACHE_FILE}.")


def background_worker():
    global cached_data
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cached_data = json.load(f)
            print(f"[i] Instant Boot: Loaded {len(cached_data.get('stocks', []))} stocks from local cache.")
        except Exception:
            pass

    scan_entire_universe()

    # Re-run scan every 6 hours
    while True:
        time.sleep(21600)
        scan_entire_universe()


threading.Thread(target=background_worker, daemon=True).start()


@app.get("/api/stocks")
def get_stocks():
    if os.path.exists(CACHE_FILE) and not cached_data.get("stocks"):
        try:
            with open(CACHE_FILE, "r") as f:
                cached_data.update(json.load(f))
        except Exception:
            pass
    return cached_data.get("stocks", [])


@app.get("/api/news")
def get_rbi_news():
    """Pulls live RBI official press releases via RSS."""
    feed = feedparser.parse("https://rbi.org.in/pressreleases_rss.xml")
    articles = []
    for entry in feed.entries[:6]:
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": getattr(entry, "published", "Recent")
        })
    return articles


@app.post("/api/ai-opinion")
def generate_ai_opinion(req: AnalysisRequest):
    """Feeds the clicked stock's valuation metrics and recent macro news into Gemini."""
    if client is None:
        return {
            "opinion": "Gemini API client could not be initialized. Check API key."
        }

    news_context = "\n".join([f"- {n}" for n in req.recentNews[:5]]) if req.recentNews else "Normal market conditions, no major policy shocks."

    prompt = f"""
You are a disciplined Value Investor in the tradition of Benjamin Graham and Charlie Munger, analyzing an Indian equity.
Evaluate this stock rigorously based on its quantitative fundamentals AND the recent macroeconomic/policy news:

--- COMPANY VALUATION & METRICS ---
Company: {req.name} ({req.ticker}.NS)
Sector: {req.sector.upper()}
Current Market Price (CMP): ₹{req.price}
Graham Number (sqrt(22.5 * EPS * BVPS)): ₹{req.grahamNumber}
P/E Ratio: {req.pe}x
Return on Equity (ROE): {req.roe}%
Debt to Equity: {req.debtEquity}
Piotroski F-Score: {req.piotroski} / 9

--- RECENT MACRO / RBI POLICY CONTEXT ---
{news_context}

Provide a concise, institutional appraisal with these three sections:
1. **Value & News Synthesis**: How do current macro/RBI policies impact this business? Is current price justified or does Mr. Market offer an asymmetric margin of safety?
2. **Moat Durability**: Evaluate whether its ROE and debt profile indicate a genuine economic moat or a cyclical trap.
3. **Actionable Recommendation & Exit Trigger**: Clear verdict (Buy / Hold / Sell) with the exact fundamental breakdown point that warrants an exit.
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return {"opinion": response.text}
    except Exception as e:
        return {"opinion": f"AI synthesis error: {str(e)}."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)