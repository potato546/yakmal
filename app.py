import json
import re
import time
from urllib.parse import unquote

import requests
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types

# 키는 코드에 넣지 않습니다.
#  - 로컬: 같은 폴더에 .streamlit/secrets.toml 파일
#  - Streamlit Cloud: 앱 설정 > Secrets 에 입력
DATA_GO_KR_KEY = st.secrets.get("DATA_GO_KR_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

MODEL = "gemini-3.6-flash"
DATA_KEY = unquote(DATA_GO_KR_KEY)

# 기본 검색: 의약품 제품 허가정보 (모든 허가 의약품 포함). 버전 번호가 바뀌면 이 줄만 수정.
PERMIT_URL = "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService07/getDrugPrdtPrmsnDtlInq06"
# 보조: e약은요 (쉬운 설명이 있는 제품만)
EASY_DRUG_URL = "https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"

# 언어명 → (BCP-47 TTS 코드, 성조/발음 특성 메모)
LANGUAGES = {
    "영어": ("en-US", "강세 위치"),
    "중국어": ("zh-CN", "4성조 + 경성"),
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

client = genai.Client(api_key=GEMINI_API_KEY)


# ---------- 식약처 API ----------
def _get(url: str, params: dict) -> list[dict]:
    r = requests.get(url, params={"serviceKey": DATA_KEY, "type": "json", "pageNo": 1, **params}, timeout=12)
    r.raise_for_status()
    return r.json().get("body", {}).get("items") or []


def search_variants(q: str) -> list[str]:
    """검색 실패 시 시도할 변형: 환→원, 마지막 제형 글자 제거, 앞 3~4글자"""
    q = q.strip()
    out = [q]
    if q.endswith("환"):
        out.append(q[:-1] + "원")
    m = re.sub(r"(정|캡슐|연질캡슐|액|시럽|산|환|원|연고|크림|겔|패치|파스)$", "", q)
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
def search_permit(q: str, rows: int = 8) -> tuple[list[dict], str]:
    """허가정보에서 제품명 검색. 결과 없으면 변형 검색어로 재시도. (결과, 실제로 쓴 검색어) 반환"""
    for v in search_variants(q):
        items = _get(PERMIT_URL, {"item_name": v, "numOfRows": rows})
        if items:
            return items, v
    return [], q


@st.cache_data(ttl=3600, show_spinner=False)
def easy_by_seq(item_seq: str) -> dict | None:
    """e약은요에 쉬운 설명이 있으면 가져오기 (없는 제품이 많음)"""
    try:
        items = _get(EASY_DRUG_URL, {"itemSeq": item_seq, "numOfRows": 1})
        return items[0] if items else None
    except Exception:
        return None


def doc_text(xml: str | None) -> str:
    """허가사항 XML(EE_DOC_DATA 등)에서 텍스트만 추출"""
    if not xml:
        return ""
    t = re.sub(r"<!\[CDATA\[|\]\]>", "", xml)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_material(material: str | None) -> list[str]:
    """'총량 : 1정|성분명 : 티아민질산염|분량 : 50|단위 : 밀리그램|...;...' → ['티아민질산염 50밀리그램', ...]"""
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


def build_source_text(permit: dict, easy: dict | None) -> str:
    lines = [
        f"[제품명] {permit.get('ITEM_NAME','')}",
        f"[제조사] {permit.get('ENTP_NAME','')}",
        f"[구분] {permit.get('ETC_OTC_CODE','')}",
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


# ---------- Gemini ----------
SYSTEM = """당신은 한국 약국에서 외국인 고객을 응대하는 약사를 돕는 도우미입니다.
반드시 아래 JSON 하나만 반환하세요. JSON 외의 텍스트, 마크다운 펜스는 금지.
원문에는 [허가사항] 전문과, 있을 경우 [쉬운설명]이 함께 옵니다. 내용은 허가사항이 기준이고, 쉬운설명은 표현을 부드럽게 하는 데 참고하세요.
사용상 주의사항 전문은 매우 길 수 있으니 일반 고객에게 실제로 중요한 것(금기, 흔한 부작용, 병용 주의)만 추리세요.

1. patient_guide: 고객에게 보여줄 복약 안내. 대상 언어로, 짧은 문장, 쉬운 표현.
   각 항목은 {"ko": 한국어, "tr": 대상 언어} 쌍. ko는 약사가 번역을 검토하기 위한 것이므로 tr과 내용이 정확히 일치해야 함.
   - what_it_is: 2~3문장. 첫 문장은 효능효과를 쉬운 말로. 이어서 주성분이 어떻게 작용해서 그 효과를 내는지
     한 문장 (일반 약학 지식, 보수적으로). 고객이 "왜 이 약이 나한테 맞는지" 이해할 수 있게.
   - how_to_take, cautions, see_pharmacist_if: 원문에 있는 내용만 근거로. 없는 정보는 지어내지 마세요.
   - ingredients: [주성분] 목록이 있을 때만. 성분별 배열 [{"name_ko":..., "name_tr":..., "amount":..., "role_ko":..., "role_tr":...}].
     name은 원료명이 아니라 고객이 알아들을 이름 (예: "빌베리건조엑스" → "빌베리 추출물(안토시아닌)").
     role은 1~2문장: 이 성분이 몸에서 무엇을 하고 이 제품에서 어떤 역할인지. 일반 약학 지식이므로 과장 없이.
     같은 계열 성분(비타민 B군, 생약 복합 등)이 여러 개면 묶어서 하나로 써도 됨. 주성분 목록이 없으면 빈 배열.

2. key_phrases: 한국인 약사가 이 제품을 팔면서 직접 입으로 말하면 효과적인 단어·짧은 구 8~9개.
   반드시 아래 세 묶음을 순서대로 포함:
   (a) 효능 3~4개: 이 제품 고유의 것. "대상 + 동사" 형태의 짧은 문장으로 (예: "야맹증을 개선해요", "망막 변성을 개선해요").
       동사는 원문(효능효과)의 표현을 따를 것 — 개선/완화/치료/예방 중 원문에 쓰인 것만. 원문이 "개선"이면 "치료"로 올리지 말 것.
       성분 1개 포함 (예: "빌베리 성분이에요").
   (b) 복용 2~3개: 이 제품의 실제 용법에서 (예: 하루 두세 번, 한 캡슐, 식후).
   (c) 확인 2개: 약사가 물어보거나 알려줄 것 (예: 당뇨 있으세요?, 2주 지나도 안 나으면 병원).
   문장 전체가 아니라 핵심 단어·짧은 구. 각 항목에 "group": "효능" | "복용" | "확인" 을 넣을 것.
   각 항목: {"group":..., "ko": 한국어, "native": 대상 언어 표기, "roman": 로마자(성조 부호 포함, 라틴문자 언어는 원문 그대로),
             "hangul": 한국인이 읽기 쉬운 한글 근사 발음, "tip": 성조/강세/발음 팁 한 줄}
   hangul은 한국어 음운으로 최대한 가깝게. tip은 한국어 화자가 틀리기 쉬운 지점을 구체적으로.

형식:
{"patient_guide": {"what_it_is": {"ko":"","tr":""}, "how_to_take": {"ko":"","tr":""}, "cautions": {"ko":"","tr":""},
                   "see_pharmacist_if": {"ko":"","tr":""}, "ingredients": []},
 "key_phrases": []}"""


@st.cache_data(ttl=3600, show_spinner=False)
def generate(source_text: str, language: str) -> dict:
    tts_code, phon = LANGUAGES[language]
    user = f"대상 언어: {language} (발음 특성: {phon})\n\n식약처 원문:\n{source_text}"
    last_err = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
            if not resp.text:
                raise RuntimeError(f"빈 응답: {resp.candidates[0].finish_reason if resp.candidates else resp}")
            raw = re.sub(r"^```(?:json)?|```$", "", resp.text.strip(), flags=re.M).strip()
            return json.loads(raw)
        except Exception as e:
            last_err = e
            if "503" in str(e) or "429" in str(e):
                time.sleep(3)
                continue
            raise
    raise last_err


# ---------- TTS 버튼 ----------
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
        return v.get("tr", ""), v.get("ko", "")
    return str(v or ""), ""


# ---------- 화면 ----------
st.set_page_config(page_title="약말", page_icon="💊", layout="wide")
st.title("💊 약말 — 외국인 복약안내 + 약사 발음 도우미")

if not DATA_GO_KR_KEY or not GEMINI_API_KEY:
    st.warning("Secrets에 DATA_GO_KR_KEY, GEMINI_API_KEY를 넣어주세요.")
    st.stop()

col_q, col_l = st.columns([3, 1])
query = col_q.text_input("제품명 검색 (예: 타이레놀, 우황청심원, 임팩타민)")
language = col_l.selectbox("고객 언어", list(LANGUAGES))

if query:
    with st.spinner("식약처 허가정보 조회 중…"):
        try:
            items, used = search_permit(query)
        except Exception as e:
            st.error(f"조회 실패: {e}")
            items, used = [], query

    if not items:
        st.info("허가정보에서 해당 제품을 찾지 못했어요. 제품명을 바꿔보세요. (박카스 등 의약외품은 이 데이터에 없습니다)")
    else:
        if used != query.strip():
            st.caption(f"'{query}'로는 결과가 없어 '{used}'로 검색했어요.")
        names = [f"{d.get('ITEM_NAME','')} ({d.get('ENTP_NAME','')})" for d in items]
        pick = st.radio("제품 선택", names, horizontal=True)
        permit = items[names.index(pick)]
        easy = easy_by_seq(permit.get("ITEM_SEQ", ""))
        source_text = build_source_text(permit, easy)

        if "전문" in str(permit.get("ETC_OTC_CODE", "")):
            st.error("전문의약품입니다. 처방전 없이 판매할 수 없어요.")
        if easy is None:
            st.caption("e약은요 쉬운 설명은 없는 제품이라 허가사항 전문으로 생성합니다.")

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
            g = out.get("patient_guide", {})
            for label, k in [
                ("이 약은", "what_it_is"),
                ("복용법", "how_to_take"),
                ("주의사항", "cautions"),
                ("약사 상담이 필요한 경우", "see_pharmacist_if"),
            ]:
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
                    st.markdown(f"- **{i.get('name_tr','')}** {i.get('amount','')} — {i.get('role_tr','')}")
                    st.caption(f"{i.get('name_ko','')} — {i.get('role_ko','')}")
                st.caption("성분·함량: 식약처 허가정보 · 역할 설명: 일반 약학 정보(AI)")
            st.caption("근거: 식약처 의약품 허가정보" + (" + e약은요" if easy else "") + " · AI 생성 안내, 약사 확인 후 제공")

        with right:
            st.subheader("🗣️ 약사가 직접 말하기")
            last_group = None
            for p in out.get("key_phrases", []):
                if p.get("group") and p.get("group") != last_group:
                    st.markdown(f"**▸ {p['group']}**")
                    last_group = p["group"]
                with st.container(border=True):
                    c1, c2 = st.columns([5, 1])
                    c1.markdown(f"**{p.get('ko','')}**")
                    c1.markdown(f"<span style='font-size:1.6em'>{p.get('native','')}</span>", unsafe_allow_html=True)
                    c1.markdown(f"`{p.get('roman','')}` · **{p.get('hangul','')}**")
                    c1.caption(f"💡 {p.get('tip','')}")
                    with c2:
                        tts_button(p.get("native", ""), tts_code)
