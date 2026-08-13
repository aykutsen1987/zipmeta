"""
aTem Central Backend
=====================
One shared backend for aTem, the common AI assistant embedded in every "Meta"
Android app (ZipMeta, and every app added to the catalog after it).

Architecture (see /README.md for the full picture):

    Android Meta App (any app_id)
          |  app_id / app_name / package_name / message
          v
        aTem  --->  Render Backend (this service)
                        |
                        v
                 CATALOG_URL (MetaCatalog Render)
                        |
                        v
             GitHub apps_catalog.json (source of truth)
                        |
                        v
                     Groq API
                        |
                        v
                  aTem structured response
                        |
                        v
                  back to the Android app

- GROQ_API_KEY lives ONLY in this server's environment variables. It is never
  sent to, or embedded in, any Android app.
- CATALOG_URL points at the single central MetaCatalog Render service. No app
  catalog data is stored locally in this backend or in any Android APK — the
  catalog is fetched (and periodically refreshed) from CATALOG_URL, so adding
  or updating a Meta app only requires editing GitHub's apps_catalog.json,
  never rebuilding or resubmitting any app.
- Exposes POST /api/atem/chat -> { "answer": "...", "recommended_apps": [...] }
- The system prompt is built per-request from the requesting app's catalog
  profile, so behavior can be tuned or a brand-new app can be onboarded
  without shipping a new APK for any existing app.
- Groq is asked to answer with structured JSON (answer + recommended_app_ids).
  This backend NEVER trusts a Play Store URL coming back from the model — it
  always re-resolves recommended_app_ids against its own catalog, so aTem can
  never invent or leak an incorrect Play Store link.

Run locally:
    uvicorn main:app --reload --port 8000

Deploy on Render:
    Build command:  pip install -r requirements.txt
    Start command:  uvicorn main:app --host 0.0.0.0 --port $PORT
    Environment:    GROQ_API_KEY=<your key>
                    GROQ_MODEL=llama-3.3-70b-versatile   (or any model you prefer)
                    CATALOG_URL=https://metacatalog-c22c.onrender.com
"""

import json
import logging
import os
import time
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq, APITimeoutError, APIStatusError, APIConnectionError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("atem-backend")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("GROQ_TIMEOUT_SECONDS", "20"))
MAX_MESSAGE_LENGTH = 2000

CATALOG_URL = os.environ.get("CATALOG_URL", "https://metacatalog-c22c.onrender.com")
CATALOG_ENDPOINT = CATALOG_URL.rstrip("/") + "/apps_catalog.json"
CATALOG_TTL_SECONDS = float(os.environ.get("CATALOG_TTL_SECONDS", "300"))
CATALOG_FETCH_TIMEOUT_SECONDS = float(os.environ.get("CATALOG_FETCH_TIMEOUT_SECONDS", "10"))

if not GROQ_API_KEY:
    # Fail loudly on startup rather than silently misbehaving in production.
    logger.warning("GROQ_API_KEY is not set. /api/atem/chat will return 500 until it is configured.")

groq_client: Optional[Groq] = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ---------------------------------------------------------------------------
# App catalog: the ONLY place that knows about every Meta app is the central
# MetaCatalog Render service at CATALOG_URL, which mirrors GitHub's
# apps_catalog.json. This backend never stores app data of its own — it
# fetches the catalog remotely and caches it in memory for CATALOG_TTL_SECONDS
# so a change on GitHub reaches every Meta app's aTem without any APK update.
# ---------------------------------------------------------------------------

_catalog_cache: dict = {}
_catalog_cache_at: float = 0.0


def _normalize_catalog_payload(payload) -> dict:
    """apps_catalog.json may come back in a few different shapes depending on
    how MetaCatalog Render serves it. Accept all of them:
      - a plain list of app objects: [ {app_id: ..., ...}, ... ]
      - a wrapper object:            { "apps": [ {app_id: ..., ...}, ... ] }
      - already keyed by app_id:     { "zipmeta": {app_id: ..., ...}, ... }
    Anything else is treated as malformed.
    """
    if isinstance(payload, dict) and "apps" in payload and isinstance(payload["apps"], list):
        payload = payload["apps"]

    if isinstance(payload, list):
        result = {}
        for app in payload:
            if isinstance(app, dict) and "app_id" in app:
                result[app["app_id"]] = app
        return result

    if isinstance(payload, dict):
        # Could already be {app_id: {...}} — verify entries look like app objects.
        result = {}
        for key, app in payload.items():
            if isinstance(app, dict):
                result[app.get("app_id", key)] = app
        return result

    return {}


def _fetch_catalog_remote() -> dict:
    try:
        with httpx.Client(timeout=CATALOG_FETCH_TIMEOUT_SECONDS) as client:
            response = client.get(CATALOG_ENDPOINT)
            response.raise_for_status()
            payload = response.json()
        catalog = _normalize_catalog_payload(payload)
        if not catalog:
            logger.error(
                "Catalog response from %s parsed to an empty/unrecognized catalog (payload type: %s)",
                CATALOG_ENDPOINT,
                type(payload).__name__,
            )
        return catalog
    except httpx.HTTPError as e:
        logger.error("Failed to fetch catalog from %s: %s", CATALOG_ENDPOINT, e)
        return {}
    except json.JSONDecodeError as e:
        logger.error("Catalog response from %s is not valid JSON: %s", CATALOG_ENDPOINT, e)
        return {}


def get_catalog() -> dict:
    """Returns the cached catalog, refreshing it from CATALOG_URL once the
    cache has expired. Falls back to the last known-good cache if a refresh
    fails, so a transient MetaCatalog Render outage doesn't take aTem down."""
    global _catalog_cache, _catalog_cache_at

    if time.monotonic() - _catalog_cache_at < CATALOG_TTL_SECONDS and _catalog_cache:
        return _catalog_cache

    fresh = _fetch_catalog_remote()
    if fresh:
        _catalog_cache = fresh
        _catalog_cache_at = time.monotonic()
        return _catalog_cache

    # Refresh failed — keep serving the stale cache (if any) rather than empty.
    return _catalog_cache


app = FastAPI(title="aTem Central Backend", version="1.1.0")

# Restrict this to your real app/package origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class RecommendedApp(BaseModel):
    app_id: str
    app_name: str
    package_name: str
    reason: str
    play_store_url: Optional[str] = None


class ChatRequest(BaseModel):
    app_id: str = Field(..., min_length=1)
    app_name: str = Field(..., min_length=1)
    package_name: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    recommended_apps: list[RecommendedApp] = []


@app.get("/")
def health_check():
    catalog = get_catalog()
    return {
        "status": "ok",
        "service": "atem-central-backend",
        "catalog_url": CATALOG_ENDPOINT,
        "apps_loaded": len(catalog),
    }


@app.get("/api/atem/apps")
def list_apps():
    """Lets a client (or a developer) sanity-check what aTem currently knows about."""
    return {"apps": list(get_catalog().values())}


def build_system_prompt(current_app: dict, catalog: dict) -> str:
    other_apps = [a for a in catalog.values() if a["app_id"] != current_app["app_id"]]
    catalog_summary = "\n".join(
        f"- app_id: {a['app_id']} | {a.get('app_name', a.get('name', ''))} — {a.get('description', '')}"
        for a in other_apps
    ) or "(şu an başka uygulama yok)"

    return f"""Sen aTem'sin.

Sen, Meta uygulamalarının ortak yapay zeka uygulama asistanısın.

Şu anda kullanıcı {current_app.get('app_name', current_app.get('name', ''))} uygulamasını kullanıyor.

Mevcut uygulamanın bilgileri:
{json.dumps(current_app, ensure_ascii=False, indent=2)}

Geliştiricinin diğer uygulamaları:
{catalog_summary}

Görevin:
- Kullanıcının mevcut uygulama hakkındaki sorularını doğru ve anlaşılır şekilde cevapla.
- Uygulamanın özelliklerini ve nasıl kullanılacağını gerektiğinde adım adım anlat.
- Kullanıcı başka bir uygulamanın yapabileceği bir işlem sorarsa ve katalogda gerçekten
  ilgili bir uygulama varsa, onu doğal ve kısa şekilde öner; neden uygun olduğunu belirt.
- Kullanıcı istemediği sürece HER cevapta başka uygulama önerme; yalnızca gerçekten
  ilgiliyse öner.
- Katalogda olmayan bir uygulama hakkında kesin bilgi uydurma.
- Cevaplarını Türkçe ver. Kısa sorulara kısa, detay istenirse ayrıntılı cevap ver.
- Kullanıcının cihazına, dosyalarına, GPS'ine veya diğer verilere erişimin yoksa
  erişimin varmış gibi davranma.
- Sen bir reklam botu değilsin; kullanıcının uygulamayı daha iyi kullanmasına
  yardımcı olan akıllı bir uygulama asistanısın.

YANIT FORMATI: Yalnızca aşağıdaki JSON şemasına uyan tek bir JSON nesnesi döndür,
başka hiçbir açıklama veya metin ekleme:
{{
  "answer": "kullanıcıya gösterilecek Türkçe cevap",
  "recommended_app_ids": ["katalogdaki app_id değerleri, ilgiliyse; yoksa boş liste"]
}}"""


def resolve_recommended_apps(app_ids: list, current_app_id: str, catalog: dict) -> list[RecommendedApp]:
    """Never trust the model's text for app identity or Play Store links —
    only IDs it returns are looked up, and only against our own catalog."""
    resolved = []
    for app_id in app_ids or []:
        if app_id == current_app_id:
            continue
        catalog_entry = catalog.get(app_id)
        if not catalog_entry:
            continue
        resolved.append(
            RecommendedApp(
                app_id=catalog_entry["app_id"],
                app_name=catalog_entry.get("app_name", catalog_entry.get("name", "")),
                package_name=catalog_entry.get("package_name", ""),
                reason=catalog_entry.get("description", ""),
                play_store_url=catalog_entry.get("play_store_url"),
            )
        )
    return resolved


@app.post("/api/atem/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if groq_client is None:
        raise HTTPException(status_code=500, detail="aTem şu anda kullanılamıyor.")

    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Mesaj boş olamaz.")

    catalog = get_catalog()

    # Fall back to a minimal profile built from the request itself if this app_id
    # hasn't been added to the catalog yet — aTem should still be able to talk
    # about the app in general terms rather than failing outright.
    current_app = catalog.get(request.app_id) or {
        "app_id": request.app_id,
        "app_name": request.app_name,
        "package_name": request.package_name,
        "description": "",
    }

    system_prompt = build_system_prompt(current_app, catalog)

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            max_tokens=1024,
            timeout=REQUEST_TIMEOUT_SECONDS,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Groq returned non-JSON content, falling back to raw text")
            parsed = {"answer": raw.strip(), "recommended_app_ids": []}

        answer = str(parsed.get("answer", "")).strip() or "Üzgünüm, şu anda bir cevap oluşturamadım."
        recommended_apps = resolve_recommended_apps(parsed.get("recommended_app_ids", []), request.app_id, catalog)

        return ChatResponse(answer=answer, recommended_apps=recommended_apps)

    except APITimeoutError:
        logger.warning("Groq request timed out")
        raise HTTPException(status_code=504, detail="Yanıt alınamadı. Lütfen tekrar dene.")
    except APIConnectionError:
        logger.warning("Could not reach Groq API")
        raise HTTPException(status_code=502, detail="aTem şu anda kullanılamıyor.")
    except APIStatusError as e:
        if e.status_code == 429:
            logger.warning("Groq rate limit hit")
            raise HTTPException(status_code=429, detail="Şu anda çok fazla istek var. Biraz sonra tekrar deneyebilirsin.")
        logger.error("Groq API error: %s", e)
        raise HTTPException(status_code=502, detail="aTem şu anda kullanılamıyor.")
    except Exception:
        # Never leak internal exception details to the client/app.
        logger.exception("Unexpected error while calling Groq")
        raise HTTPException(status_code=500, detail="aTem şu anda kullanılamıyor.")
