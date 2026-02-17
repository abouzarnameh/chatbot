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

# NEW: Image model (default gpt-image-1.5)
AVALAI_IMAGE_MODEL = os.getenv("AVALAI_IMAGE_MODEL", "gpt-image-1.5").strip()

SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE", "").strip()
SYSTEM_PROMPT_ENV = os.getenv("SYSTEM_PROMPT", "").strip()

AVALAI_BASE_URL = "https://api.avalai.ir/v1"

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


# -------------------- NEW: AvalAI Image Generation --------------------
async def avalai_generate_image(prompt: str, size: str = "1024x1024", timeout_s: float = 120.0) -> str:
    headers = {
        "Authorization": f"Bearer {AVALAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": AVALAI_IMAGE_MODEL,  # gpt-image-1.5
        "prompt": prompt,
        "size": size,
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(f"{AVALAI_BASE_URL}/images/generations", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    # expected: {"data":[{"url":"..."}]}
    return data["data"][0]["url"]


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


# -------------------- Handlers --------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"سلام! من {BOT_CALL_NAME} هستم.\n"
        "تو گروه: یا اسممو اول پیام بیار، یا منشن کن، یا روی پیام من ریپلای بزن.\n"
        "برای عکس هم: /image توضیح عکس"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    history[uid].clear()
    await update.message.reply_text("حافظه گفتگو پاک شد.")


# NEW: /image command
async def image_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    prompt = " ".join(context.args).strip() if context.args else ""
    if not prompt:
        await update.message.reply_text("مثال: /image یه خرس کارتونی با عینک و کت")
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")

    try:
        url = await avalai_generate_image(prompt=prompt, size="1024x1024")
        await update.message.reply_photo(photo=url)
    except Exception as e:
        await update.message.reply_text(f"خطا در ساخت تصویر: {type(e).__name__}: {str(e)[:200]}")


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

    # NEW: /image
    app.add_handler(CommandHandler("image", image_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
