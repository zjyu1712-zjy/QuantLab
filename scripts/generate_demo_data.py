"""Generate deterministic, fictional ETF/trade CSVs; no external data."""
from pathlib import Path
from datetime import date, timedelta
import csv
import math

root = Path(__file__).resolve().parents[1] / "examples"
root.mkdir(exist_ok=True)
rows = []
day = date(2024, 1, 1)
while len(rows) < 260:
    if day.weekday() < 5:
        i = len(rows)
        price = 4 + 0.002*i + 0.18*math.sin(i/12)
        rows.append([day.isoformat(), "510300", round(price, 4), round(price+0.04, 4),
                     round(price-0.04, 4), round(price+0.01*math.cos(i), 4), 100000+i*100])
    day += timedelta(days=1)
with (root / "synthetic_etf.csv").open("w", newline="") as f:
    w=csv.writer(f, lineterminator="\n"); w.writerow(["date","symbol","open","high","low","close","volume"]); w.writerows(rows)
with (root / "synthetic_trades.csv").open("w", newline="") as f:
    w=csv.writer(f, lineterminator="\n"); w.writerow(["trade_id","trade_datetime","symbol","side","quantity","price","fee"])
    for n, i in enumerate([10, 80]):
        w.writerow([f"DEMO{n+1}", rows[i][0]+" 10:00:00", "510300", "buy" if n == 0 else "sell", 100, rows[i][2], 0.5])
print("Generated fictional examples in", root)
