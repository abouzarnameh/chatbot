import os
import re
import asyncio
import threading
from collections import defaultdict, deque

import httpx
from flask import Flask
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

load_dotenv()

# -------------------- ENV --------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
AVALAI_API_KEY = os.getenv("AVALAI_API_KEY", "").strip()
AVALAI_MODEL = os.getenv("AVALAI_MODEL", "gpt-4o-mini").strip()
BOT_CALL_NAME = os.getenv("BOT_CALL_NAME", "سس خرسی").strip()

SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE", "").strip()
SYSTEM_PROMPT_ENV = os.getenv("SYSTEM_PROMPT", "").strip()

BRSAPI_KEY = os.getenv("BRSAPI_KEY", "").strip()

AVALAI_BASE_URL = "https://api.avalai.ir/v1"
BRSAPI_FREE_URL = "https://brsapi.ir/Api/Market/Gold_Currency.php"

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
if not AVALAI_API_KEY:
    raise RuntimeError("Missing AVALAI_API_KEY")


def load_system_prompt() -> str:
    if SYSTEM_PROMPT_FILE:
        path = SYSTEM_PROMPT_FILE
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(__file__), path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            if txt:
                return txt
        except Exception:
            pass

    if SYSTEM_PROMPT_ENV:
        return SYSTEM_PROMPT_ENV

    return "You are a helpful assistant."


SYSTEM_PROMPT = load_system_prompt()

# -------------------- Flask (برای Render Web Service) --------------------
web = Flask(__name__)


@web.get("/")
def home():
    return "OK"


def run_web():
    port = int(os.environ.get("PORT", "10000"))
    web.run(host="0.0.0.0", port=port, use_reloader=False)


# -------------------- Conversation memory (per user) --------------------
MAX_TURNS = 10
history = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))


# -------------------- AvalAI Chat --------------------
async def avalai_chat(messages: list[dict], timeout_s: float = 60.0) -> str:
    headers = {
        "Authorization": f"Bearer {AVALAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": AVALAI_MODEL, "messages": messages, "temperature": 0.7}

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(f"{AVALAI_BASE_URL}/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    return (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip()


# -------------------- Text normalization (fix invisible chars / spacing) --------------------
def _norm(s: str) -> str:
    if not s:
        return ""
    # common invisible chars in Telegram / Persian texts
    s = s.replace("\u200c", "")  # ZWNJ
    s = s.replace("\u200f", "")  # RLM
    s = s.replace("\u200e", "")  # LRM
    s = s.replace("\ufeff", "")  # BOM
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


# -------------------- Group Reply Rules (Call/Mention/Reply) --------------------
def should_reply_in_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    msg = update.message
    if not msg or not msg.text:
        return False

    text = _norm(msg.text)
    bot_username = _norm(context.bot.username or "")
    call_name = _norm(BOT_CALL_NAME)
    call_name_compact = call_name.replace(" ", "")

    # 1) Call name at beginning (normal/compact)
    if text.startswith(call_name) or text.replace(" ", "").startswith(call_name_compact):
        return True

    # 2) Mention
    if bot_username and f"@{bot_username}" in text:
        return True

    # 3) Reply to bot message
    if msg.reply_to_message and msg.reply_to_message.from_user:
        if msg.reply_to_message.from_user.id == context.bot.id:
            return True

    return False


def clean_user_text(text: str, bot_username: str) -> str:
    raw = (text or "").strip()
    t_norm = _norm(raw)

    bn = _norm(bot_username or "")
    cn = _norm(BOT_CALL_NAME)
    cn_compact = cn.replace(" ", "")

    # remove call name at beginning (best-effort)
    if t_norm.startswith(cn):
        raw = raw[len(BOT_CALL_NAME):].strip()
    else:
        # compact form: "سسخرسی سلام" / with invisible chars
        compact = t_norm.replace(" ", "")
        if compact.startswith(cn_compact):
            parts = raw.split(maxsplit=1)
            raw = parts[1].strip() if len(parts) > 1 else ""

    # remove mention
    if bn:
        raw = raw.replace(f"@{bot_username}", "").replace(f"@{(bot_username or '').lower()}", "").strip()

    return raw.strip()


# -------------------- BRSAPI (Gold/Currency) --------------------
PRICE_KEYWORDS = {
    "قیمت", "نرخ", "چنده", "چند", "الان", "لحظه", "لحظه‌ای", "امروز",
    "دلار", "یورو", "پوند", "درهم", "لیر", "یوان",
    "سکه", "امامی", "بهار", "نیم", "ربع", "گرمی",
    "طلا", "طلای", "۱۸", "18", "24", "اونس", "انس",
    "ارز", "تتر", "btc", "bitcoin", "بیت", "بیتکوین",
}


def looks_like_price_request(user_text: str) -> bool:
    txt = _norm(user_text)
    return bool(txt) and any(k in txt for k in PRICE_KEYWORDS)


def normalize_query_for_price(user_text: str) -> str:
    txt = (user_text or "").strip()
    txt = re.sub(r"[^\w\u0600-\u06FF\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


async def fetch_brsapi_items() -> list[dict]:
    if not BRSAPI_KEY:
        raise RuntimeError("BRSAPI_KEY is not set")
    params = {"key": BRSAPI_KEY}
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.get(BRSAPI_FREE_URL, params=params)
        r.raise_for_status()
        data = r.json()

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("data", "result", "items"):
            if k in data and isinstance(data[k], list):
                return data[k]
    return []


def find_best_matches(items: list[dict], q: str, max_n: int = 6) -> list[dict]:
    ql = _norm(q)
    tokens = [t for t in ql.split() if t and t not in {"قیمت", "نرخ", "الان", "امروز", "لحظه", "لحظه‌ای", "چنده", "چند"}]

    sym_guess = None
    m = re.search(r"\b([A-Za-z]{3,6})\b", q)
    if m:
        sym_guess = m.group(1).upper()

    scored: list[tuple[int, dict]] = []
    for it in items:
        name = str(it.get("name", ""))
        name_en = str(it.get("name_en", ""))
        symbol = str(it.get("symbol", ""))
        hay = f"{name} {name_en} {symbol}".lower()

        score = 0
        if sym_guess and symbol.upper() == sym_guess:
            score += 50
        for tok in tokens:
            if tok in hay:
                score += 10

        if "دلار" in ql and ("usd" in symbol.lower() or "دلار" in name):
            score += 8
        if "سکه" in ql and ("سکه" in name):
            score += 8
        if ("طلا" in ql or "۱۸" in ql or "18" in ql) and ("طلا" in name or "gold" in name_en.lower()):
            score += 6

        if score > 0:
            scored.append((score, it))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:max_n]]


def format_price_lines(items: list[dict]) -> str:
    dt_str = None
    if items:
        d = items[0].get("date")
        t = items[0].get("time")
        if d and t:
            dt_str = f"{d} {t}"

    lines = []
    for it in items:
        name = it.get("name") or it.get("name_en") or it.get("symbol") or "—"
        symbol = it.get("symbol")
        price = it.get("price")
        unit = it.get("unit") or ""
        chv = it.get("change_value")
        chp = it.get("change_percent")

        change_part = ""
        if chp is not None:
            change_part = f" | تغییر: {chp}%"
        elif chv is not None:
            change_part = f" | تغییر: {chv}"

        sym_part = f" ({symbol})" if symbol else ""
        unit_part = f" {unit}" if unit else ""
        lines.append(f"- {name}{sym_part}: {price}{unit_part}{change_part}")

    header = f"قیمت‌های لحظه‌ای (BRSAPI){' — ' + dt_str if dt_str else ''}:"
    return header + "\n" + "\n".join(lines)


async def send_prices(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str) -> None:
    if not BRSAPI_KEY:
        await update.message.reply_text("کلید BRSAPI_KEY ست نشده. تو Render توی Env Vars اضافه‌اش کن.")
        return
    try:
        items = await fetch_brsapi_items()
        if not items:
            await update.message.reply_text("از API دیتا نگرفتم یا خروجی خالی بود.")
            return

        qn = normalize_query_for_price(query)
        matches = find_best_matches(items, qn, max_n=6)
        if not matches:
            matches = items[:6]

        await update.message.reply_text(format_price_lines(matches))
    except Exception as e:
        await update.message.reply_text(f"خطا در گرفتن قیمت: {type(e).__name__}: {str(e)[:200]}")


# -------------------- Handlers --------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"سلام! من {BOT_CALL_NAME} هستم.\n"
        "تو گروه: یا اسممو اول پیام بیار، یا منشن کن، یا روی پیام من ریپلای بزن."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    history[uid].clear()
    await update.message.reply_text("حافظه گفتگو پاک شد.")


async def price_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = " ".join(context.args).strip() if context.args else ""
    if not q:
        q = "دلار طلا سکه"
    await send_prices(update, context, q)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    chat_type = update.effective_chat.type
    bot_username = context.bot.username or ""

    # Group rules: call/mention/reply
    if chat_type in ["group", "supergroup"]:
        if not should_reply_in_group(update, context):
            return

    user_text = clean_user_text(text, bot_username)
    if not user_text:
        return

    # Price request: direct BRSAPI
    if looks_like_price_request(user_text):
        await send_prices(update, context, user_text)
        return

    uid = update.effective_user.id

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(list(history[uid]))
    msgs.append({"role": "user", "content": user_text})

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        answer = await avalai_chat(msgs)
        if not answer:
            answer = "پاسخی دریافت نشد."
    except Exception as e:
        answer = f"خطا: {type(e).__name__}: {str(e)[:200]}"

    history[uid].append({"role": "user", "content": user_text})
    history[uid].append({"role": "assistant", "content": answer})

    await update.message.reply_text(answer)


def main() -> None:
    # Python 3.14: create event loop explicitly
    asyncio.set_event_loop(asyncio.new_event_loop())

    # Web server on separate daemon thread (Render Web Service)
    threading.Thread(target=run_web, daemon=True).start()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("price", price_cmd))  # /price دلار
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
