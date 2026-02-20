# backend/app/services/ai_service.py

import uuid
import time
import random
import google.generativeai as genai
from fastapi import HTTPException
from ..config import settings

# --- 1. GEMINI API BAŞLATMA ---
try:
    genai.configure(api_key=settings.GEMINI_API_KEY)
except Exception as e:
    print(f"HATA: Gemini API yapılandırılamadı: {e}")

# Modelleri Tanımla
flash_model = genai.GenerativeModel("models/gemini-flash-latest")
pro_model = genai.GenerativeModel("models/gemini-pro-latest")


# ==========================================
# Session Store (PDF Sohbet Hafızası)
# ==========================================
# session_id -> { text, filename, history:[{role, content}], created_at, llm_provider, mode }
_PDF_CHAT_SESSIONS = {}
SESSION_TTL_SECONDS = 60 * 60  # 1 saat

def _cleanup_sessions():
    """Süresi dolan sohbet oturumlarını temizler."""
    now = time.time()
    expired = [
        sid for sid, s in _PDF_CHAT_SESSIONS.items()
        if (now - s["created_at"]) > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _PDF_CHAT_SESSIONS[sid]


# ==========================================
# Yardımcı Fonksiyonlar (Retry & Error Handling)
# ==========================================

def _is_quota_or_rate_limit_error(err: Exception) -> bool:
    """Hatanın kota veya hız limitiyle ilgili olup olmadığını anlar."""
    msg = str(err)
    return ("429" in msg) or ("Quota exceeded" in msg) or ("rate limit" in msg.lower())


def _generate_with_retry(model, prompt: str, attempts: int = 5):
    """
    API çağrısını yapar. 429 hatası alırsa bekleyip tekrar dener.
    """
    last_err = None
    for i in range(attempts):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            last_err = e
            if _is_quota_or_rate_limit_error(e):
                sleep_s = min(60, (2 ** i)) + random.random() * 0.5
                print(f"⚠️ Gemini Rate Limit ({i+1}/{attempts}). {sleep_s:.2f}s bekleniyor...")
                time.sleep(sleep_s)
                continue
            raise 
    raise last_err


# ==========================================
# Ana Servis Fonksiyonları
# ==========================================

def gemini_generate(text_content: str, prompt_instruction: str, mode: str = "flash") -> str:
    if not text_content:
        raise HTTPException(status_code=400, detail="Boş içerik gönderildi.")

    MAX_TEXT_LENGTH = 50000
    if len(text_content) > MAX_TEXT_LENGTH:
        text_content = text_content[:MAX_TEXT_LENGTH]

    full_prompt = f"{prompt_instruction}\n\nMETİN:\n---\n{text_content}\n---"
    model = flash_model if mode == "flash" else pro_model

    try:
        response = _generate_with_retry(model, full_prompt, attempts=3)
        if getattr(response, "candidates", None):
            return response.text

        raise HTTPException(
            status_code=400, 
            detail="AI'dan geçerli bir yanıt alınamadı (içerik engellenmiş olabilir)."
        )
    except HTTPException:
        raise
    except Exception as e:
        if _is_quota_or_rate_limit_error(e):
            raise HTTPException(status_code=429, detail=f"Gemini servis yoğunluğu: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gemini servisinde hata: {str(e)}")


def call_gemini_for_task(text_content: str, prompt_instruction: str) -> str:
    """Celery (Arka Plan) görevleri için kullanılır."""
    if not text_content or not text_content.strip():
        raise HTTPException(status_code=400, detail="Boş içerik gönderildi.")

    MAX_TEXT_LENGTH = 50000
    if len(text_content) > MAX_TEXT_LENGTH:
        text_content = text_content[:MAX_TEXT_LENGTH]

    full_prompt = f"{prompt_instruction}\n\nMETİN:\n---\n{text_content}\n---"

    # 1. Deneme: PRO Modeli
    try:
        resp = _generate_with_retry(pro_model, full_prompt, attempts=4)
        if getattr(resp, "candidates", None):
            return resp.text
    except Exception as e:
        if not _is_quota_or_rate_limit_error(e):
            raise e
        print("⚠️ Gemini Pro kotası dolu, Flash modeline geçiliyor...")

    # 2. Deneme (Fallback): FLASH Modeli
    try:
        resp = _generate_with_retry(flash_model, full_prompt, attempts=5)
        if getattr(resp, "candidates", None):
            return resp.text
        raise HTTPException(status_code=400, detail="AI yanıt üretmedi (flash fallback).")
    except Exception as e:
        if _is_quota_or_rate_limit_error(e):
            raise HTTPException(status_code=429, detail=f"Gemini tamamen dolu: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gemini task hatası: {str(e)}")


# ==========================================
# PDF Chat Fonksiyonları
# ==========================================

# DÜZELTME: Router'dan gelen 'llm_provider' ve 'mode' parametreleri buraya eklendi.
def create_pdf_chat_session(
    pdf_text: str, 
    filename: str | None = None, 
    llm_provider: str = "cloud", 
    mode: str = "flash"
) -> str:
    """Yeni bir sohbet oturumu başlatır ve ID döner."""
    _cleanup_sessions()
    session_id = str(uuid.uuid4())

    _PDF_CHAT_SESSIONS[session_id] = {
        "text": pdf_text,
        "filename": filename or "uploaded.pdf",
        "history": [],
        "created_at": time.time(),
        "llm_provider": llm_provider,
        "mode": mode,                 
        "qa_cache": {} # YENİ: Bu oturumdaki aynı soruları yakalamak için cache
    }
    return session_id


def chat_with_pdf(session_id: str, user_message: str) -> str:
    """
    NOT: Bu fonksiyon ai_service içindeki local Gemini chat fonksiyonudur.
    Router tarafında 'chat_over_pdf' (llm_manager) kullanılıyorsa bu kullanılmayabilir,
    ancak legacy destek veya fallback için burada tutulmuştur.
    """
    _cleanup_sessions()

    session = _PDF_CHAT_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sohbet oturumu bulunamadı veya süresi dolmuş.")

    if not user_message or not user_message.strip():
        raise HTTPException(status_code=400, detail="Mesaj boş olamaz.")

    pdf_text = session["text"]
    filename = session["filename"]
    history = session["history"]

    MAX_CONTEXT_CHARS = 45000
    pdf_context = pdf_text[:MAX_CONTEXT_CHARS] if len(pdf_text) > MAX_CONTEXT_CHARS else pdf_text

    system_instruction = (
        "Sen bir PDF asistanısın. Kullanıcının yüklediği PDF'e dayanarak cevap ver.\n"
        "Eğer PDF'te açıkça yoksa, bunu belirt ve kullanıcıdan sayfa/başlık gibi ipucu iste.\n"
        "Cevaplarını Türkçe ver, net ve pratik ol.\n"
    )

    history_text = ""
    for turn in history[-10:]:
        history_text += f"{turn['role'].upper()}: {turn['content']}\n"

    prompt = f"""
{system_instruction}

DOSYA: {filename}

PDF İÇERİĞİ:
---
{pdf_context}
---

SOHBET GEÇMİŞİ:
---
{history_text}
---

KULLANICI SORUSU:
{user_message}
""".strip()

    try:
        try:
            response = _generate_with_retry(pro_model, prompt, attempts=3)
        except Exception as e:
            if _is_quota_or_rate_limit_error(e):
                response = _generate_with_retry(flash_model, prompt, attempts=3)
            else:
                raise

        if not getattr(response, "candidates", None):
            raise HTTPException(status_code=400, detail="AI yanıt üretmedi.")

        answer = response.text

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": answer})

        return answer

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sohbet hatası: {str(e)}")