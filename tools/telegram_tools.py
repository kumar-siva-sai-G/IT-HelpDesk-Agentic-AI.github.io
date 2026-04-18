"""
tools/telegram_tools.py
Telegram Bot API integration.

Uses python-telegram-bot for:
- Sending messages to users
- Receiving incoming tickets
- Sending admin alerts
"""

import asyncio
from loguru import logger

from config.settings import TELEGRAM_BOT_TOKEN


def send_telegram_message(
    chat_id: str,
    text: str,
    parse_mode: str = None,
) -> bool:
    """
    Send a message to a Telegram chat.

    Args:
        chat_id: Telegram chat ID
        text: Message text
        parse_mode: "Markdown" or "HTML" (optional)

    Returns:
        True on success, False on failure
    """
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("[TELEGRAM] No bot token configured. Message not sent.")
        logger.info(f"[TELEGRAM MOCK] → chat_id={chat_id}: {text[:100]}")
        return False

    if not chat_id or chat_id == "":
        logger.warning("[TELEGRAM] No chat_id provided. Message not sent.")
        return False

    try:
        import httpx
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode

        response = httpx.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.debug(f"[TELEGRAM] Message sent to {chat_id}")
            return True
        else:
            logger.error(f"[TELEGRAM] Failed: {response.status_code} {response.text}")
            return False

    except Exception as e:
        logger.error(f"[TELEGRAM] Error: {e}")
        return False


async def start_telegram_bot(on_message_callback):
    """
    Start the Telegram bot polling loop.
    Calls on_message_callback(chat_id, text) for each new message.

    Uses manual initialize/start/start_polling to work within an existing
    asyncio event loop (run_polling() tries to manage its own loop).

    Args:
        on_message_callback: async function(chat_id: str, text: str)
    """
    if not TELEGRAM_BOT_TOKEN:
        logger.error("[TELEGRAM] Cannot start bot: TELEGRAM_BOT_TOKEN not set")
        return

    from telegram import Update
    from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message and update.message.text:
            chat_id = str(update.message.chat_id)
            text = update.message.text
            logger.info(f"[TELEGRAM] Incoming from {chat_id}: {text[:80]}")
            await on_message_callback(chat_id, text)

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("[TELEGRAM] Bot started. Listening for messages...")

    # Manual lifecycle management to work within an existing event loop
    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    # Keep running until cancelled
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
