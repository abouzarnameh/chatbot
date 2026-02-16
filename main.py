import os
from collections import defaultdict, deque

import httpx
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
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")
if not AVALAI_API_KEY:
    raise RuntimeError("Missing AVALAI_API_KEY in .env")


def load_system_prompt() -> str:
    # اولویت: فایل prompt.txt
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

    # بعدی: از env
    if SYSTEM_PROMPT_ENV:
        return SYSTEM_PROMPT_ENV

    # پیشفرض
    return "You are a helpful assistant."


SYSTEM_PROMPT = load_system_prompt()

MAX_TURNS = 10
history = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))


async def avalai_chat(messages: list[dict], timeout_s: float = 60.0) -> str:
    headers = {
        "Authorization": f"Bearer {AVALAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": AVALAI_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }

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
    if not text:
        return ""
    t = text.strip()

    if t.lower().startswith(BOT_CALL_NAME.lower()):
        t = t[len(BOT_CALL_NAME):].strip()

    if bot_username:
        t = t.replace(f"@{bot_username}", "").replace(f"@{bot_username.lower()}", "").strip()

    return t


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"سلام! من {BOT_CALL_NAME} هستم 🐻🍯\n"
        "تو گروه فقط وقتی اسممو صدا بزنی جواب میدم.\n"
        f"مثال: {BOT_CALL_NAME} سلام"
    )


async def show_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    sys_prompt = context.bot_data.get("SYSTEM_PROMPT", SYSTEM_PROMPT)
    await update.message.reply_text(f"System Prompt فعلی:\n\n{sys_prompt[:3500]}")


async def set_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text or ""
    parts = text.split(" ", 1)
    if len(parts) < 2 or not parts[1].strip():
        await update.message.reply_text("استفاده: /setprompt <متن پرامپت>")
        return

    context.bot_data["SYSTEM_PROMPT"] = parts[1].strip()
    await update.message.reply_text("✅ پرامپت سیستم (برای همین اجرا) تنظیم شد.")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    history[uid].clear()
    await update.message.reply_text("🧹 حافظه گفتگو پاک شد.")


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
    sys_prompt = context.bot_data.get("SYSTEM_PROMPT", SYSTEM_PROMPT)

    msgs = [{"role": "system", "content": sys_prompt}]
    msgs.extend(list(history[uid]))
    msgs.append({"role": "user", "content": user_text})

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        answer = await avalai_chat(msgs)
        if not answer:
            answer = "پاسخی دریافت نشد."
    except httpx.HTTPStatusError as e:
        answer = f"خطای API: {e.response.status_code}\n{e.response.text[:400]}"
    except Exception as e:
        answer = f"خطا: {type(e).__name__}: {str(e)[:300]}"

    history[uid].append({"role": "user", "content": user_text})
    history[uid].append({"role": "assistant", "content": answer})

    await update.message.reply_text(answer)


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("prompt", show_prompt))
    app.add_handler(CommandHandler("setprompt", set_prompt))
    app.add_handler(CommandHandler("reset", reset))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
