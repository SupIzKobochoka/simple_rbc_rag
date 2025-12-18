import asyncio
import html
import logging
import re
from dataclasses import dataclass
from typing import List, Tuple

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from API_KEY import API_KEY_TELEGRAM
from main_rag import get_rag_responce


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("rag_bot")


@dataclass
class TokenBlock:
    token: str
    kind: str
    a: str = ""
    b: str = ""


def _normalize_lists(md: str) -> str:
    out = []
    for line in (md or "").replace("\r\n", "\n").split("\n"):
        m = re.match(r"^\s*([*\-+])\s+(.*)$", line)
        if m:
            out.append("• " + m.group(2))
            continue
        m = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
        if m:
            out.append(f"{m.group(1)}) {m.group(2)}")
            continue
        out.append(line)
    return "\n".join(out)


def _extract_fenced_code(md: str) -> Tuple[str, List[TokenBlock]]:
    blocks: List[TokenBlock] = []

    def repl(m):
        lang = (m.group(1) or "").strip()
        code = m.group(2) or ""
        token = f"§§CODE{len(blocks)}§§"
        blocks.append(TokenBlock(token=token, kind="codeblock", a=lang, b=code))
        return token

    md2 = re.sub(r"```([^\n]*)\n(.*?)```", repl, md, flags=re.DOTALL)
    return md2, blocks


def _extract_inline_code(md: str) -> Tuple[str, List[TokenBlock]]:
    blocks: List[TokenBlock] = []

    def repl(m):
        code = m.group(1) or ""
        token = f"§§INCODE{len(blocks)}§§"
        blocks.append(TokenBlock(token=token, kind="incode", a=code))
        return token

    md2 = re.sub(r"`([^`\n]+)`", repl, md)
    return md2, blocks


def _extract_links(md: str) -> Tuple[str, List[TokenBlock]]:
    blocks: List[TokenBlock] = []

    def repl(m):
        text = m.group(1) or ""
        url = m.group(2) or ""
        token = f"§§LINK{len(blocks)}§§"
        blocks.append(TokenBlock(token=token, kind="link", a=text, b=url))
        return token

    md2 = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl, md)
    return md2, blocks


def _restore_tokens(escaped_text: str, tokens: List[TokenBlock]) -> str:
    text = escaped_text

    for t in tokens:
        if t.kind == "link":
            safe_txt = html.escape(t.a)
            safe_url = html.escape(t.b, quote=True)
            text = text.replace(html.escape(t.token), f'<a href="{safe_url}">{safe_txt}</a>')

    for t in tokens:
        if t.kind == "incode":
            safe = html.escape(t.a)
            text = text.replace(html.escape(t.token), f"<code>{safe}</code>")

    for t in tokens:
        if t.kind == "codeblock":
            safe = html.escape(t.b)
            text = text.replace(html.escape(t.token), f"<pre><code>{safe}</code></pre>")

    return text


def _apply_emphasis(escaped_text: str) -> str:
    t = escaped_text
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t, flags=re.DOTALL)

    def ital(m):
        inner = (m.group(1) or "").strip()
        return f"<i>{inner}</i>" if inner else m.group(0)

    t = re.sub(r"(?<!\*)\*(\s*[^*\n]+?\s*)\*(?!\*)", ital, t)
    t = re.sub(r"(?<!_)_(\s*[^_\n]+?\s*)_(?!_)", ital, t)

    return t


def md_to_tg_html(md: str) -> str:
    md = _normalize_lists(md or "")

    md, code_blocks = _extract_fenced_code(md)
    md, inline_codes = _extract_inline_code(md)
    md, links = _extract_links(md)

    tokens: List[TokenBlock] = []
    tokens.extend(code_blocks)
    tokens.extend(inline_codes)
    tokens.extend(links)

    escaped = html.escape(md)
    emphasized = _apply_emphasis(escaped)
    restored = _restore_tokens(emphasized, tokens)

    return restored


def _split_md_blocks(md: str) -> List[str]:
    md = (md or "").replace("\r\n", "\n")
    lines = md.split("\n")

    blocks: List[str] = []
    cur: List[str] = []
    in_code = False

    for line in lines:
        if line.strip().startswith("```"):
            cur.append(line)
            in_code = not in_code
            continue

        if not in_code and line.strip() == "":
            if cur:
                blocks.append("\n".join(cur).strip("\n"))
                cur = []
            continue

        cur.append(line)

    if cur:
        blocks.append("\n".join(cur).strip("\n"))

    return [b for b in blocks if b.strip()]


def chunk_for_telegram(md: str, limit: int = 3800) -> List[str]:
    parts: List[str] = []
    blocks = _split_md_blocks(md)

    cur_md = ""
    for b in blocks:
        candidate = b if not cur_md else (cur_md + "\n\n" + b)
        if len(md_to_tg_html(candidate)) <= limit:
            cur_md = candidate
            continue

        if cur_md:
            parts.append(cur_md)
            cur_md = ""

        if len(md_to_tg_html(b)) <= limit:
            cur_md = b
            continue

        acc = ""
        for line in b.split("\n"):
            cand2 = line if not acc else (acc + "\n" + line)
            if len(md_to_tg_html(cand2)) <= limit:
                acc = cand2
            else:
                if acc.strip():
                    parts.append(acc)
                acc = line
        if acc.strip():
            parts.append(acc)

    if cur_md.strip():
        parts.append(cur_md)

    return parts if parts else [""]


async def send_rag(update: Update, question: str):
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not question or not question.strip():
        return

    q = question.strip()
    log.info(f"IN | chat_id={chat.id} | user={user.username or user.id} | q={q}")

    await chat.send_action(ChatAction.TYPING)

    try:
        md = await asyncio.to_thread(get_rag_responce, q)
    except Exception as e:
        log.exception("RAG error")
        await chat.send_message(f"Ошибка при получении ответа: {e}")
        return

    md = md or ""
    for part_md in chunk_for_telegram(md, limit=3800):
        html_msg = md_to_tg_html(part_md)
        await chat.send_message(html_msg, parse_mode="HTML", disable_web_page_preview=False)


async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = " ".join(context.args) if context.args else ""
    if not question.strip():
        await update.message.reply_text("Формат: /ask твой вопрос")
        return
    await send_rag(update, question)


def main():
    if not API_KEY_TELEGRAM or not isinstance(API_KEY_TELEGRAM, str):
        raise RuntimeError("API_KEY_TELEGRAM не задан или не строка")

    app = Application.builder().token(API_KEY_TELEGRAM).build()

    app.add_handler(CommandHandler("ask", ask_cmd))

    app.run_polling()


if __name__ == "__main__":
    main()
