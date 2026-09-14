import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote

import requests
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
from gtts import gTTS

# 키는 코드에 넣지 않습니다.
#  - 로컬: 같은 폴더에 .streamlit/secrets.toml 파일
#  - Streamlit Cloud: 앱 설정 > Secrets 에 입력
DATA_GO_KR_KEY = st.secrets.get("DATA_GO_KR_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
NAVER_CLIENT_ID = st.secrets.get("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = st.secrets.get("NAVER_CLIENT_SECRET", "")
NAVER_BASE = "https://naverapihub.apigw.ntruss.com/search/v1"  # NAVER API HUB (네이버클라우드)

MODEL = "gemini-3.1-flash-lite"  # 텍스트 생성용 (무료 한도 넉넉)
MODEL_MEDIA = "gemini-3.6-flash"  # 음성·사진 이해용 (더 정확, 무료 하루 20회. 한도 넘으면 자동으로 위 모델로 전환)
DATA_KEY = unquote(DATA_GO_KR_KEY)

# 식약처 API (버전 번호가 바뀌면 이 줄들만 수정)
PERMIT_URL = "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService07/getDrugPrdtPrmsnDtlInq06"  # 의약품 허가정보
EASY_DRUG_URL = "https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"        # e약은요 (보조)
HTFS_URL = "https://apis.data.go.kr/1471000/HtfsInfoService03/getHtfsItem01"                        # 건강기능식품
COSM_URL = "https://apis.data.go.kr/1471000/FtnltCosmRptPrdlstInfoService01/getRptPrdlstInq01"        # 기능성화장품 보고품목
COSM_SRNG_URL = "https://apis.data.go.kr/1471057/FtnltCosmSrngPrdlstInfoService05/getSrngPrdlstInq05"  # 기능성화장품 심사품목

# 언어명 → (BCP-47 TTS 코드, 성조/발음 특성 메모)
LANGUAGES = {
    "영어": ("en-US", "강세 위치"),
    "중국어(보통화)": ("zh-CN", "4성조 + 경성, 간체자"),
    "중국어(번체·대만)": ("zh-TW", "4성조, 번체자, 대만 표현"),
    "광둥어": ("zh-HK", "6성조(Jyutping 숫자 성조), 번체자, 홍콩 구어체"),
    "일본어": ("ja-JP", "장음·촉음·고저 악센트"),
    "베트남어": ("vi-VN", "6성조"),
    "태국어": ("th-TH", "5성조"),
    "인도네시아어": ("id-ID", "강세 약함, 모음 명확"),
    "아랍어": ("ar-SA", "인후음·강조자음"),
    "터키어": ("tr-TR", "모음조화, 마지막 음절 강세"),
    "몽골어": ("mn-MN", "장모음 구분"),
    "프랑스어": ("fr-FR", "비모음, 연음"),
    "독일어": ("de-DE", "첫 음절 강세, ch 발음"),
    "이탈리아어": ("it-IT", "이중자음, 모음 명확"),
    "러시아어": ("ru-RU", "강세 위치에 따른 모음 약화"),
    "스페인어": ("es-ES", "r 굴림, 뒤에서 두 번째 음절 강세"),
}

# gTTS(구글 음성) 언어 코드. 없는 언어(광둥어·몽골어)는 브라우저 음성으로 폴백
GTTS_LANG = {
    "영어": "en", "중국어(보통화)": "zh-CN", "중국어(번체·대만)": "zh-TW", "일본어": "ja", "베트남어": "vi",
    "태국어": "th", "인도네시아어": "id", "아랍어": "ar", "터키어": "tr", "프랑스어": "fr", "독일어": "de",
    "이탈리아어": "it", "러시아어": "ru", "스페인어": "es",
}

client = genai.Client(api_key=GEMINI_API_KEY)


# ---------- 식약처 API ----------
def _get(url: str, params: dict) -> list[dict]:
    r = requests.get(url, params={"serviceKey": DATA_KEY, "type": "json", "pageNo": 1, **params}, timeout=12)
    r.raise_for_status()
    items = r.json().get("body", {}).get("items") or []
    # 건기식 API는 {"item": {...}} 로 한 겹 더 싸여 있음
    return [it.get("item", it) for it in items]


def search_variants(q: str) -> list[str]:
    """검색 실패 시 시도할 변형: 환→원, 제형 글자 제거, 앞 3~4글자"""
    q = q.strip()
    out = [q]
    if q.endswith("환"):
        out.append(q[:-1] + "원")
    m = re.sub(r"(정|캡슐|연질캡슐|액|시럽|산|환|원|연고|크림|겔|패치|파스|스틱|분말)$", "", q)
    if m and m != q:
        out.append(m)
    if len(q) > 4:
        out.append(q[:4])
    if len(q) > 3:
        out.append(q[:3])
    seen, uniq = set(), []
    for v in out:
        if v and v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


@st.cache_data(ttl=3600, show_spinner=False)
def search_drug(q: str, rows: int = 6) -> list[dict]:
    for v in search_variants(q):
        items = _get(PERMIT_URL, {"item_name": v, "numOfRows": rows})
        if items:
            return [{"kind": "의약품", "name": it.get("ITEM_NAME", ""), "maker": it.get("ENTP_NAME", ""), "raw": it} for it in items]
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def search_htfs(q: str, rows: int = 6) -> list[dict]:
    for v in search_variants(q):
        items = _get(HTFS_URL, {"Prduct": v, "numOfRows": rows})
        if items:
            return [{"kind": "건강기능식품", "name": it.get("PRDUCT", "").strip(), "maker": it.get("ENTRPS", ""), "raw": it} for it in items]
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def search_cosm(q: str, rows: int = 8) -> list[dict]:
    """기능성화장품: 보고품목 + 심사품목 둘 다 조회"""
    for v in search_variants(q):
        out, seen = [], set()
        try:
            rpt = [it for it in _get(COSM_URL, {"item_name": v, "numOfRows": rows}) if it.get("CANCEL_APPROVAL_YN") != "Y"]
        except Exception:
            rpt = []
        for it in rpt:
            it["_src"] = "보고"
            seen.add(it.get("ITEM_NAME", "").replace(" ", ""))
            out.append(it)
        try:
            srng = _get(COSM_SRNG_URL, {"item_name": v, "numOfRows": rows})
        except Exception:
            srng = []
        for it in srng:
            nm = it.get("ITEM_NAME", "").replace(" ", "")
            if nm and nm not in seen:
                it["_src"] = "심사"
                out.append(it)
        if out:
            return [{"kind": "기능성화장품", "name": it.get("ITEM_NAME", ""), "maker": it.get("ENTP_NAME", ""), "raw": it} for it in out]
    return []


def search_all(q: str) -> tuple[list[dict], dict]:
    """의약품·건기식·기능성화장품 동시 검색. (결과, 소스별 오류) 반환"""
    errors = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {"의약품": ex.submit(search_drug, q), "건강기능식품": ex.submit(search_htfs, q),
                "기능성화장품": ex.submit(search_cosm, q)}
    results = []
    for kind, f in futs.items():
        try:
            results += f.result()
        except Exception as e:
            errors[kind] = str(e)
    return results, errors


@st.cache_data(ttl=3600, show_spinner=False)
def easy_by_seq(item_seq: str) -> dict | None:
    try:
        items = _get(EASY_DRUG_URL, {"itemSeq": item_seq, "numOfRows": 1})
        return items[0] if items else None
    except Exception:
        return None


def doc_text(xml: str | None) -> str:
    if not xml:
        return ""
    t = re.sub(r"<!\[CDATA\[|\]\]>", "", xml)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_material(material: str | None) -> list[str]:
    out = []
    for chunk in (material or "").split(";"):
        kv = {}
        for part in chunk.split("|"):
            if ":" in part:
                k, v = part.split(":", 1)
                kv[k.strip()] = v.strip()
        name = kv.get("성분명")
        if name:
            amt = f"{kv.get('분량', '')}{kv.get('단위', '')}".strip()
            out.append(f"{name} {amt}".strip())
    return out


def strip_html(s: str | None) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def clean(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def build_drug_text(permit: dict, easy: dict | None) -> str:
    lines = [
        f"[제품군] 의약품 — {permit.get('ETC_OTC_CODE','')}",
        f"[제품명] {permit.get('ITEM_NAME','')}",
        f"[제조사] {permit.get('ENTP_NAME','')}",
    ]
    ingr = parse_material(permit.get("MATERIAL_NAME"))
    if ingr:
        lines.append("[주성분(허가정보)] " + ", ".join(ingr))
    for label, key, limit in [("효능효과(허가사항)", "EE_DOC_DATA", 1500),
                              ("용법용량(허가사항)", "UD_DOC_DATA", 1200),
                              ("사용상 주의사항(허가사항)", "NB_DOC_DATA", 2500)]:
        t = doc_text(permit.get(key))
        if t:
            lines.append(f"[{label}] {t[:limit]}")
    if easy:
        for label, key in [("효능(쉬운설명)", "efcyQesitm"), ("용법(쉬운설명)", "useMethodQesitm"),
                           ("주의사항(쉬운설명)", "atpnQesitm"), ("상호작용(쉬운설명)", "intrcQesitm"),
                           ("부작용(쉬운설명)", "seQesitm")]:
            t = strip_html(easy.get(key))
            if t:
                lines.append(f"[{label}] {t}")
    return "\n".join(lines)


def build_htfs_text(h: dict) -> str:
    lines = [
        "[제품군] 건강기능식품 (의약품 아님 — 질병 치료·예방 표현 금지, 식약처 인정 기능성 문구 '~에 도움을 줄 수 있음'만 사용)",
        f"[제품명] {clean(h.get('PRDUCT'))}",
        f"[제조사] {clean(h.get('ENTRPS'))}",
    ]
    for label, key in [("기능성 내용(식약처 인정)", "MAIN_FNCTN"), ("섭취량·섭취방법", "SRV_USE"),
                       ("섭취 시 주의사항", "INTAKE_HINT1"), ("성상", "SUNGSANG"),
                       ("보관방법", "PRSRV_PD"), ("유통기한", "DISTB_PD"), ("기준규격", "BASE_STANDARD")]:
        t = clean(h.get(key))
        if t:
            lines.append(f"[{label}] {t[:1200]}")
    return "\n".join(lines)



# ---------- 네이버 검색 (전성분·제형·후기) ----------
def naver_ok() -> bool:
    return bool(NAVER_CLIENT_ID and NAVER_CLIENT_SECRET)


@st.cache_data(ttl=86400, show_spinner=False)
def naver_search(kind: str, query: str, display: int = 20) -> list[dict]:
    """kind: blog | webkr | cafearticle | news"""
    r = requests.get(f"{NAVER_BASE}/{kind}", params={"query": query, "display": display, "format": "json"},
                     headers={"X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID, "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET}, timeout=12)
    r.raise_for_status()
    return r.json().get("items") or []


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_text(url: str, limit: int = 6000) -> str:
    """페이지 본문 텍스트 (네이버 블로그는 모바일 주소로)"""
    try:
        u = url.replace("https://blog.naver.com/", "https://m.blog.naver.com/")
        r = requests.get(u, timeout=10, headers={"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"})
        html = r.text
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"&nbsp;|&amp;|&lt;|&gt;|&quot;", " ", text)
        return re.sub(r"\s+", " ", text).strip()[:limit]
    except Exception:
        return ""


NAVER_INGR_COMMON = """당신은 한국 약국에서 외국인 고객을 응대하는 약사를 돕는 도우미입니다.
반드시 JSON 하나만 반환하세요. JSON 외의 텍스트, 마크다운 펜스는 금지.
원문 첫 줄의 [제품군]을 반드시 확인하세요.
- 의약품: 효능효과 원문의 동사(개선/완화/치료/예방)를 그대로 따르고, 원문이 "개선"이면 "치료"로 올리지 말 것.
- 건강기능식품: 절대 "치료", "예방", "효과가 있다"라고 하지 말고, 식약처 인정 기능성 문구대로 "~에 도움을 줄 수 있어요"만 사용.
  "복용" 대신 "섭취", "약" 대신 "건강기능식품/영양제"라고 부를 것.
- 기능성화장품: 식약처 보고 기능성(미백/주름개선/자외선차단 등)을 "~에 도움을 줘요"로만. 치료·질환 표현 금지. '복용' 금지, '바르다/사용하다'.
원문에는 [허가사항] 전문과, 있을 경우 [쉬운설명]이 함께 옵니다. 내용은 허가사항이 기준이고, 쉬운설명은 표현 참고용.
"""

SYSTEM_GUIDE = COMMON + """
patient_guide: 고객에게 보여줄 안내. 대상 언어로, 짧은 문장, 쉬운 표현.
각 항목은 {"ko": 한국어, "tr": 대상 언어} 쌍. ko는 약사가 번역을 검토하기 위한 것이므로 tr과 내용이 정확히 일치해야 함.
- what_it_is: 2~3문장. 첫 문장은 효능(또는 기능성)을 쉬운 말로. 이어서 주성분이 어떻게 작용하는지 한 문장 (일반 약학 지식, 보수적으로).
- how_to_take: 용법·섭취방법·사용법. cautions: 일반 고객에게 실제로 중요한 것(금기, 흔한 부작용, 병용 주의)만.
  see_pharmacist_if. 모두 원문에 있는 내용만 근거로. 없는 정보는 지어내지 마세요.
- ingredients: 주성분·원료·전성분 정보가 있을 때만. [{"name_ko":..., "name_tr":..., "amount":..., "role_ko":..., "role_tr":...}].
  name은 고객이 알아들을 이름. role은 1~2문장, 과장 없이. 화장품은 핵심 역할 성분 3~5개만. 정보가 없으면 빈 배열. 반드시 객체 배열로.
형식: {"what_it_is": {"ko":"","tr":""}, "how_to_take": {"ko":"","tr":""}, "cautions": {"ko":"","tr":""},
       "see_pharmacist_if": {"ko":"","tr":""}, "ingredients": []}"""

SYSTEM_PHRASES = COMMON + """
key_phrases: 한국인 약사가 이 제품을 팔면서 직접 입으로 말하면 효과적인 단어·짧은 구 8~9개.
반드시 아래 세 묶음을 순서대로 포함:
(a) 효능 3~4개: 이 제품 고유의 것. "대상 + 동사" 짧은 문장 (의약품: "야맹증을 개선해요" / 건기식: "면역력에 도움을 줄 수 있어요"
    / 화장품: "미백에 도움을 줘요", "SPF 50이에요"). 성분 1개 포함 (예: "홍삼 성분이에요").
(b) 복용 2~3개: 실제 용법·섭취방법·사용법에서 (예: 하루 두세 번, 한 캡슐, 식후, 아침저녁, 마지막 단계에).
(c) 확인 2개: 약사가 물어보거나 알려줄 것 (예: 당뇨약 드세요?, 2주 지나도 안 나으면 병원, 민감성 피부세요?).
각 항목에 "group": "효능" | "복용" | "확인".
각 항목: {"group":..., "ko": 한국어, "native": 대상 언어 표기, "roman": 로마자(성조 부호 포함, 라틴문자 언어는 원문 그대로),
          "hangul": 한국인이 읽기 쉬운 한글 근사 발음, "tip": 성조/강세/발음 팁 한 줄}
광둥어는 native를 번체자 홍콩 구어체로, roman은 Jyutping(예: gam2 mou6 joek6)으로.
hangul은 한국어 음운으로 최대한 가깝게. tip은 한국어 화자가 틀리기 쉬운 지점을 구체적으로.
형식: {"key_phrases": []}"""


def _gen_json(system: str, user: str) -> dict:
    last_err = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=user,
                config=types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                                   thinking_config=types.ThinkingConfig(thinking_level="low")),
            )
            if not resp.text:
                raise RuntimeError(f"빈 응답: {resp.candidates[0].finish_reason if resp.candidates else resp}")
            raw = re.sub(r"^```(?:json)?|```$", "", resp.text.strip(), flags=re.M).strip()
            raw = raw[raw.find("{"):]
            obj, _ = json.JSONDecoder().raw_decode(raw)
            return obj
        except Exception as e:
            last_err = e
            if "503" in str(e) or ("429" in str(e) and "per_day" not in str(e).lower()):
                time.sleep(3)
                continue
            raise
    raise last_err


@st.cache_data(ttl=3600, show_spinner=False)
def generate(source_text: str, language: str) -> dict:
    """안내문과 발음 카드를 두 요청으로 나눠 동시에 생성 (속도)"""
    tts_code, phon = LANGUAGES[language]
    user = f"대상 언어: {language} (발음 특성: {phon})\n\n식약처 원문:\n{source_text}"
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_guide = ex.submit(_gen_json, SYSTEM_GUIDE, user)
        f_phr = ex.submit(_gen_json, SYSTEM_PHRASES, user)
        guide = f_guide.result()
        phr = f_phr.result()
    return {"patient_guide": guide, "key_phrases": phr.get("key_phrases", [])}


# ---------- 음성 ----------
@st.cache_data(ttl=86400, show_spinner=False)
def tts_mp3(text: str, gtts_lang: str) -> bytes | None:
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=gtts_lang, slow=False).write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


def make_all_audio(texts: list[str], gtts_lang: str) -> list[bytes | None]:
    with ThreadPoolExecutor(max_workers=6) as ex:
        return list(ex.map(lambda t: tts_mp3(t, gtts_lang), texts))


def tts_button(text: str, lang_code: str):
    safe = json.dumps(text)
    components.html(
        f"""
        <button onclick='(function(){{
            const u = new SpeechSynthesisUtterance({safe});
            u.lang = "{lang_code}"; u.rate = 0.85;
            speechSynthesis.cancel(); speechSynthesis.speak(u);
        }})()' style="font-size:20px;border:none;background:none;cursor:pointer">🔊</button>
        """,
        height=40,
    )


def pair(v) -> tuple[str, str]:
    if isinstance(v, dict):
        return str(v.get("tr", "")), str(v.get("ko", ""))
    return str(v or ""), ""


def render_phrases(phrases: list[dict], language: str):
    phrases = [p for p in (phrases or []) if isinstance(p, dict)]
    tts_code = LANGUAGES.get(language, ("en-US", ""))[0]
    gl = GTTS_LANG.get(language)
    audios = make_all_audio([str(p.get("native", "")) for p in phrases], gl) if gl else [None] * len(phrases)
    last_group = None
    for p, audio in zip(phrases, audios):
        if p.get("group") and p.get("group") != last_group:
            st.markdown(f"**▸ {p['group']}**")
            last_group = p["group"]
        with st.container(border=True):
            st.markdown(f"**{p.get('ko','')}**")
            st.markdown(f"<span style='font-size:1.6em'>{p.get('native','')}</span>", unsafe_allow_html=True)
            st.markdown(f"`{p.get('roman','')}` · **{p.get('hangul','')}**")
            st.caption(f"💡 {p.get('tip','')}")
            if audio:
                st.audio(audio, format="audio/mp3")
            else:
                tts_button(str(p.get("native", "")), tts_code)


# ---------- 듣기 모드 (고객 음성 → 이해 + 답할 말) ----------
LISTEN_SYSTEM = """외국인 고객이 한국 약국에서 말한 음성입니다. 약사가 이해하고 바로 답할 수 있게 정리하세요.
JSON만 반환 (마크다운 펜스 금지):
{"language": "아래 목록 중 하나, 없으면 '기타'", 
 "transcript": "고객이 말한 원문 (그 언어 그대로)",
 "korean": "한국어 번역",
 "summary_ko": "고객이 원하는 것 한 문장 (증상, 찾는 제품, 질문)",
 "search_terms": ["제품 검색에 쓸 한국어 키워드 1~3개 (증상명 또는 제품군, 예: 설사, 지사제, 두통약)"],
 "reply_phrases": [ 약사가 지금 바로 그 언어로 말하면 좋은 단어·짧은 구 5~6개.
   (a) 확인 2개: 증상 되묻기·기간·다른 약 복용 여부 (예: 언제부터요?, 다른 약 드세요?)
   (b) 안내 2개: 상황에 맞는 일반 안내 (예: 잠시만요, 이 약이 있어요, 하루 세 번)
   (c) 공감 1~2개: (예: 많이 힘드셨겠어요, 괜찮아질 거예요)
   각 항목: {"group": "확인"|"안내"|"공감", "ko": 한국어, "native": 그 언어 표기, "roman": 로마자(성조 부호 포함),
            "hangul": 한글 근사 발음, "tip": 발음 팁 한 줄} ]}
언어 목록: """ + ", ".join(LANGUAGES.keys()) + """
광둥어는 native를 번체자 홍콩 구어체로, roman은 Jyutping으로. 음성이 한국어면 language를 '한국어'로 하고 reply_phrases는 빈 배열."""


def listen_and_reply(data: bytes, mime: str, hint_lang: str, is_image: bool = False) -> dict:
    """고객 음성 또는 고객이 보여준 화면/사진 → 이해 + 답할 말. 좋은 모델 먼저, 한도 걸리면 Lite로."""
    if is_image:
        ask = (f"고객이 보여준 화면/사진입니다 (번역 앱 화면, 메모, 제품 사진, 자기 나라 약 사진 등). "
               f"사진 속 글이나 제품이 무엇인지 파악해 transcript에는 보이는 글(또는 제품 설명)을, korean에는 한국어로 정리. "
               f"고객 언어 힌트: {hint_lang}")
    else:
        ask = f"이 음성을 듣고 JSON으로 정리. 고객 언어 힌트: {hint_lang} (다른 언어면 실제 언어로)."
    last = None
    for m in (MODEL_MEDIA, MODEL):
        try:
            resp = client.models.generate_content(
                model=m,
                contents=[types.Part.from_bytes(data=data, mime_type=mime), ask],
                config=types.GenerateContentConfig(
                    system_instruction=LISTEN_SYSTEM,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
            raw = re.sub(r"^```(?:json)?|```$", "", (resp.text or "").strip(), flags=re.M).strip()
            raw = raw[raw.find("{"):]
            obj, _ = json.JSONDecoder().raw_decode(raw)
            obj["_model"] = m
            return obj
        except Exception as e:
            last = e
            if "429" in str(e) or "404" in str(e):
                continue
            raise
    raise last


# ---------- 화면 ----------
st.set_page_config(page_title="약말", page_icon="💊", layout="wide")
st.title("💊 약말 — 외국인 복약안내 + 약사 발음 도우미")

if not DATA_GO_KR_KEY or not GEMINI_API_KEY:
    st.warning("Secrets에 DATA_GO_KR_KEY, GEMINI_API_KEY를 넣어주세요.")
    st.stop()

col_m, col_l = st.columns([3, 1])
mode = col_m.radio("모드", ["제품 검색", "🎤 고객 말 듣기"], horizontal=True, label_visibility="collapsed")
language = col_l.selectbox("고객 언어", list(LANGUAGES))

if mode == "🎤 고객 말 듣기":
    st.caption("고객이 말하면 녹음하거나, 고객이 보여주는 화면·사진을 찍으세요. 무슨 말인지와 바로 답할 발음 카드가 나옵니다. "
               "오른쪽 위 '고객 언어'를 맞춰두면 인식이 더 정확해요.")
    t_rec, t_img = st.tabs(["🎤 녹음", "📷 고객이 보여준 화면·사진"])
    with t_rec:
        rec = st.audio_input("녹음")
    with t_img:
        up = st.file_uploader("사진 파일 선택 (갤러리·스크린샷)", type=["jpg", "jpeg", "png", "webp"])
        cam = st.camera_input("촬영") if st.toggle("📷 카메라 켜기", key="cam_listen") else None
    img = cam or up
    src = None
    if img is not None:
        src = (img.getvalue(), img.type or "image/jpeg", True)
    elif rec is not None:
        src = (rec.getvalue(), rec.type or "audio/wav", False)
    if src:
        with st.spinner("이해하는 중…"):
            try:
                L = listen_and_reply(src[0], src[1], language, is_image=src[2])
            except Exception as e:
                st.error(f"인식 실패: {e}")
                st.stop()
        if L.get("_model") == MODEL:
            st.caption("(정확도 높은 모델 한도 초과로 기본 모델로 처리했어요)")
        lang_detected = L.get("language", "기타")
        st.markdown(f"**감지 언어:** {lang_detected}")
        st.markdown(f"**고객 말:** {L.get('transcript','')}")
        st.markdown(f"**한국어:** {L.get('korean','')}")
        st.info(f"원하는 것: {L.get('summary_ko','')}")
        terms = [t for t in (L.get("search_terms") or []) if isinstance(t, str)]
        if terms:
            st.caption("제품 검색 키워드: " + " · ".join(terms) + "  → 위에서 '제품 검색' 모드로 바꿔 검색하세요")
        st.subheader("🗣️ 지금 바로 답하기")
        lang_for_cards = lang_detected if lang_detected in LANGUAGES else language
        render_phrases(L.get("reply_phrases"), lang_for_cards)
    st.stop()

query = st.text_input("제품명 검색 (예: 타이레놀, 우황청심원, 홍삼정, 리쥬올)")

if query:
    with st.spinner("식약처 조회 중 (의약품 + 건강기능식품 + 기능성화장품)…"):
        results, errors = search_all(query)
    for kind, msg in errors.items():
        st.caption(f"⚠️ {kind} 조회 실패: {msg[:120]}")

    if not results:
        st.info("찾지 못했어요. 제품명을 바꿔보세요. (의약외품, 기능성 표시가 없는 일반 화장품은 식약처 제품 데이터가 없습니다)")
    else:
        labels = [f"[{r['kind']}] {r['name']} ({r['maker']})" for r in results]
        pick = st.radio("제품 선택", labels)
        chosen = results[labels.index(pick)]
        raw = chosen["raw"]

        if chosen["kind"] == "의약품":
            easy = easy_by_seq(raw.get("ITEM_SEQ", ""))
            source_text = build_drug_text(raw, easy)
            basis = "식약처 의약품 허가정보" + (" + e약은요" if easy else "")
            if "전문" in str(raw.get("ETC_OTC_CODE", "")):
                st.error("전문의약품입니다. 처방전 없이 판매할 수 없어요.")
        elif chosen["kind"] == "건강기능식품":
            easy = None
            source_text = build_htfs_text(raw)
            basis = "식약처 건강기능식품정보"
            st.info("건강기능식품 — 안내문과 발음 카드는 '~에 도움을 줄 수 있어요' 표현으로 생성됩니다.")
        else:
            easy = None
            fn = clean(raw.get("EE_NAME")) or doc_text(raw.get("EE_DOC_DATA")) or "기능성화장품"
            st.info(f"기능성화장품 (식약처 {raw.get('_src','보고')}) — {fn}")
            st.caption("화장품 성분·함량은 식약처 데이터에 없습니다. 파일 등록 → 사진 → 붙여넣기 순으로 찾습니다.")
            ingr_text = file_ingredients(chosen["name"])
            src_label = ""
            if ingr_text:
                src_label = " + 등록 파일 전성분"
                st.success("등록된 전성분을 사용합니다 (cosmetic_ingredients.txt)")
            else:
                if naver_ok() and st.button("🔎 네이버에서 전성분·제형 찾기", key=f"nv_{chosen['name']}"):
                    st.session_state["nv_for"] = chosen["name"]
                if naver_ok() and st.session_state.get("nv_for") == chosen["name"]:
                    with st.spinner("네이버 검색 + 페이지 읽는 중…"):
                        try:
                            w = naver_ingredients(chosen["name"], chosen["maker"])
                        except Exception as e:
                            w = {"found": False, "note": str(e)[:200]}
                    if w.get("found") and w.get("ingredients"):
                        ingr_text = w["ingredients"]
                        if w.get("disclosed_amounts"):
                            ingr_text += f"\n[회사 공개 함량] {w['disclosed_amounts']}"
                        if w.get("texture"):
                            ingr_text += f"\n[제형·사용감(웹)] {w['texture']}"
                        tag = "주요 성분 일부" if w.get("partial") else "전성분"
                        src_label = f" + {tag}(웹: {w.get('source','')}) — 포장과 대조 필요"
                        st.warning(f"웹에서 찾은 {tag} — 포장과 대조해 확인하세요. 출처: {w.get('source','')}\n\n"
                                   f"{w['ingredients'][:400]}{'…' if len(w['ingredients'])>400 else ''}"
                                   + (f"\n\n제형·사용감: {w['texture']}" if w.get('texture') else "")
                                   + (f"\n\n{w['note']}" if w.get('note') else ""))
                        st.code(f"{chosen['name']} | {w['ingredients']}", language=None)
                        st.caption("↑ 확인 후 이 줄을 cosmetic_ingredients.txt에 넣어두면 다음부턴 자동")
                    else:
                        st.caption(f"네이버에서 성분을 찾지 못했어요. {w.get('note','')} 사진이나 붙여넣기로 넣어주세요.")
                upload = st.file_uploader("전성분 사진 파일 선택", type=["jpg", "jpeg", "png", "webp"])
                photo = st.camera_input("포장의 전성분 부분을 촬영") if st.toggle("📷 카메라 켜기", key="cam_cosm") else None
                img = photo or upload
                if img is not None and not ingr_text:
                    with st.spinner("사진에서 전성분 읽는 중…"):
                        try:
                            r = read_ingredients_from_photo(img.getvalue(), img.type or "image/jpeg")
                        except Exception as e:
                            r = {"found": False, "note": str(e)[:200]}
                    if r.get("found") and r.get("ingredients"):
                        ingr_text = r["ingredients"]
                        src_label = " + 포장 촬영 전성분"
                        st.info(f"사진에서 읽은 전성분 — 포장과 대조해 확인하세요. {r.get('note','')}\n\n{ingr_text[:400]}{'…' if len(ingr_text)>400 else ''}")
                        st.code(f"{chosen['name']} | {ingr_text}", language=None)
                        st.caption("↑ 이 줄을 cosmetic_ingredients.txt에 넣어두면 다음부턴 촬영 없이 자동으로 씁니다.")
                    else:
                        st.warning(f"전성분을 읽지 못했어요. {r.get('note','')} 전성분 글자가 크게 나오도록 다시 찍거나 아래에 붙여넣어 주세요.")
                pasted = st.text_area("또는 전성분 붙여넣기", height=70, placeholder="포장의 전성분표를 여기에 붙여넣기")
                if pasted.strip():
                    ingr_text = pasted
                    src_label = " + 붙여넣은 전성분"
            source_text = build_cosm_text(raw, ingr_text)
            basis = f"식약처 기능성화장품 {raw.get('_src','보고')}품목정보" + src_label

        with st.expander("식약처 원문 보기"):
            st.text(source_text)

        with st.spinner(f"{language} 안내 + 발음 카드 생성 중…"):
            try:
                out = generate(source_text, language)
            except Exception as e:
                st.error(f"생성 실패: {e}")
                st.stop()

        tts_code = LANGUAGES[language][0]
        left, right = st.columns(2)

        with left:
            st.subheader(f"🧾 고객용 안내 ({language})")
            if chosen["kind"] == "의약품" and easy and easy.get("itemImage"):
                st.image(easy["itemImage"], width=220, caption="낱알 모양")
            g = out.get("patient_guide", {}) or {}
            head = "이 약은" if chosen["kind"] == "의약품" else "이 제품은"
            how = {"의약품": "복용법", "건강기능식품": "섭취방법"}.get(chosen["kind"], "사용법")
            for label, k in [(head, "what_it_is"), (how, "how_to_take"), ("주의사항", "cautions"),
                             ("약사 상담이 필요한 경우", "see_pharmacist_if")]:
                tr, ko = pair(g.get(k))
                if not tr:
                    continue
                st.markdown(f"**{label}**")
                st.write(tr)
                if ko:
                    st.caption(ko)

            ingr = g.get("ingredients") or []
            if ingr:
                st.markdown("**주요 성분**")
                for i in ingr:
                    if isinstance(i, dict):
                        st.markdown(f"- **{i.get('name_tr','')}** {i.get('amount','')} — {i.get('role_tr','')}")
                        st.caption(f"{i.get('name_ko','')} — {i.get('role_ko','')}")
                    else:
                        st.markdown(f"- {i}")
                st.caption("성분·함량: 식약처 데이터 · 역할 설명: 일반 약학 정보(AI)")
            st.caption(f"근거: {basis} · AI 생성 안내, 약사 확인 후 제공")

        with right:
            st.subheader("🗣️ 약사가 직접 말하기")
            render_phrases(out.get("key_phrases"), language)

        render_reviews(chosen["name"], chosen["kind"])
