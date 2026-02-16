import os
import asyncio
import threading
from collections import defaultdict, deque

import httpx
from flask import Flask
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
AVALAI_API_KEY = os.getenv("AVALAI_API_KEY", "").strip()
AVALAI_MODEL = os.getenv("AVALAI_MODEL", "gpt-4o-mini").strip()
BOT_CALL_NAME = os.getenv("BOT_CALL_NAME", "سس خرسی").strip()

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

# ------------------ Web server (برای Render Web Service) ------------------
web = Flask(__name__)

@web.get("/")
def home():
    return "OK"

def run_web():
    port = int(os.environ.get("PORT", "10000"))
    # reloader خاموش تا دوبار اجرا نشه
    web.run(host="0.0.0.0", port=port, use_reloader=False)


# ------------------ Telegram Bot ------------------
MAX_TURNS = 10
history = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))


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


def should_reply_in_group(text: str, bot_username: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    call_name = BOT_CALL_NAME.lower()

    if t.startswith(call_name):
        return True
    if bot_username and f"@{bot_username.lower()}" in t:
        return True
    return False


def clean_user_text(text: str, bot_username: str) -> str:
    t = (text or "").strip()
    if t.lower().startswith(BOT_CALL_NAME.lower()):
        t = t[len(BOT_CALL_NAME):].strip()

    if bot_username:
        t = t.replace(f"@{bot_username}", "").replace(f"@{bot_username.lower()}", "").strip()

    return t


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"سلام! من {BOT_CALL_NAME} هستم.\n"
        "تو گروه فقط وقتی اسممو صدا بزنی جواب میدم."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    history[uid].clear()
    await update.message.reply_text("حافظه گفتگو پاک شد.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    chat_type = update.effective_chat.type
    bot_username = context.bot.username or ""

    if chat_type in ["group", "supergroup"]:
        if not should_reply_in_group(text, bot_username):
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


def should_reply_in_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    message = update.message
    if not message or not message.text:
        return False

    text = message.text.strip().lower()
    bot_username = (context.bot.username or "").lower()
    call_name = BOT_CALL_NAME.lower()

    # 1) Call name
    if text.startswith(call_name):
        return True

    # 2) Mention
    if bot_username and f"@{bot_username}" in text:
        return True

    # 3) Reply to bot message
    if message.reply_to_message:
        if message.reply_to_message.from_user:
            if message.reply_to_message.from_user.id == context.bot.id:
                return True

    return False

def main() -> None:
    # برای Python 3.14: لوپ رو دستی بساز
    asyncio.set_event_loop(asyncio.new_event_loop())

    # وب‌سرور در ترد جدا (daemon تا گیر نکنه)
    threading.Thread(target=run_web, daemon=True).start()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

