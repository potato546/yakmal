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

EASY_DRUG_URL = "https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"
# 제품허가정보는 서비스 버전 번호가 가끔 바뀝니다. 공공데이터포털 페이지의 End Point와
# 오퍼레이션명이 다르면 아래 두 줄만 그 값으로 바꾸세요.
PERMIT_URL = "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService07/getDrugPrdtPrmsnDtlInq06"

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
@st.cache_data(ttl=3600, show_spinner=False)
def search_easy_drug(name: str, rows: int = 5) -> list[dict]:
    params = {"serviceKey": DATA_KEY, "itemName": name, "type": "json", "pageNo": 1, "numOfRows": rows}
    r = requests.get(EASY_DRUG_URL, params=params, timeout=10)
    r.raise_for_status()
    return r.json().get("body", {}).get("items") or []


@st.cache_data(ttl=3600, show_spinner=False)
def search_permit(item_name: str) -> dict | None:
    """제품허가정보에서 주성분·전문/일반 구분 조회. 실패하면 None (앱은 계속 동작)."""
    params = {"serviceKey": DATA_KEY, "item_name": item_name, "type": "json", "pageNo": 1, "numOfRows": 3}
    try:
        r = requests.get(PERMIT_URL, params=params, timeout=10)
        r.raise_for_status()
        items = r.json().get("body", {}).get("items") or []
    except Exception as e:
        st.session_state["permit_err"] = str(e)
        raise
    # 이름이 정확히 같은 것 우선, 없으면 첫 번째
    for it in items:
        if it.get("ITEM_NAME") == item_name:
            return it
    return items[0] if items else None


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


def drug_to_text(d: dict, permit: dict | None) -> str:
    fields = [
        ("제품명", "itemName"),
        ("제조사", "entpName"),
        ("효능", "efcyQesitm"),
        ("용법용량", "useMethodQesitm"),
        ("주의사항(경고)", "atpnWarnQesitm"),
        ("주의사항", "atpnQesitm"),
        ("상호작용", "intrcQesitm"),
        ("부작용", "seQesitm"),
        ("보관법", "depositMethodQesitm"),
    ]
    lines = [f"[{k}] {strip_html(d.get(v))}" for k, v in fields if d.get(v)]
    if permit:
        ingr = parse_material(permit.get("MATERIAL_NAME"))
        if ingr:
            lines.append("[주성분(식약처 허가정보)] " + ", ".join(ingr))
        if permit.get("ETC_OTC_CODE"):
            lines.append(f"[구분] {permit['ETC_OTC_CODE']}")
    return "\n".join(lines)


# ---------- Gemini ----------
SYSTEM = """당신은 한국 약국에서 외국인 고객을 응대하는 약사를 돕는 도우미입니다.
반드시 아래 JSON 하나만 반환하세요. JSON 외의 텍스트, 마크다운 펜스는 금지.

1. patient_guide: 고객에게 보여줄 복약 안내. 대상 언어로, 짧은 문장, 쉬운 표현.
   각 항목은 {"ko": 한국어, "tr": 대상 언어} 쌍. ko는 약사가 번역을 검토하기 위한 것이므로 tr과 내용이 정확히 일치해야 함.
   - what_it_is, how_to_take, cautions, see_pharmacist_if: 제공된 식약처 원문에 있는 내용만 근거로. 없는 정보는 지어내지 마세요.
   - ingredients: [주성분] 목록이 있을 때만. 성분별 배열 [{"name_ko":..., "name_tr":..., "amount":..., "role_ko":..., "role_tr":...}].
     role은 그 성분이 몸에서 하는 역할 한 줄 (예: 비타민 B1 — 에너지 대사, 피로 회복). 이 설명은 일반 약학 지식이므로
     과장 없이 보수적으로. 주성분 목록이 없으면 빈 배열.

2. key_phrases: 한국인 약사가 이 제품을 팔면서 직접 입으로 말하면 효과적인 단어·짧은 구 6~7개.
   문장 전체가 아니라 핵심 단어 (예: 감기약, 하루 세 번, 식후, 졸릴 수 있어요, 증상 있어요?).
   성분이 특징인 제품(비타민, 영양제 등)이면 대표 성분·효과 단어를 1~2개 포함.
   각 항목: {"ko": 한국어, "native": 대상 언어 표기, "roman": 로마자(성조 부호 포함, 라틴문자 언어는 원문 그대로),
             "hangul": 한국인이 읽기 쉬운 한글 근사 발음, "tip": 성조/강세/발음 팁 한 줄}
   hangul은 한국어 음운으로 최대한 가깝게. tip은 한국어 화자가 틀리기 쉬운 지점을 구체적으로.

형식:
{"patient_guide": {"what_it_is": {"ko":"","tr":""}, "how_to_take": {"ko":"","tr":""}, "cautions": {"ko":"","tr":""},
                   "see_pharmacist_if": {"ko":"","tr":""}, "ingredients": []},
 "key_phrases": []}"""


@st.cache_data(ttl=3600, show_spinner=False)
def generate(drug_text: str, language: str) -> dict:
    tts_code, phon = LANGUAGES[language]
    user = f"대상 언어: {language} (발음 특성: {phon})\n\n식약처 원문:\n{drug_text}"
    last_err = None
    for attempt in range(3):  # 503 등 일시 장애면 잠깐 쉬고 재시도
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
    """{"ko":..,"tr":..} 또는 문자열 → (대상언어, 한국어)"""
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
query = col_q.text_input("제품명 검색 (예: 타이레놀, 판콜, 임팩타민)")
language = col_l.selectbox("고객 언어", list(LANGUAGES))

if query:
    with st.spinner("식약처 조회 중…"):
        try:
            items = search_easy_drug(query)
        except Exception as e:
            st.error(f"조회 실패: {e}")
            items = []

    if not items:
        st.info("e약은요에서 해당 제품을 찾지 못했어요. 제품명을 바꿔보세요.")
    else:
        names = [f"{d['itemName']} ({d['entpName']})" for d in items]
        pick = st.radio("제품 선택", names, horizontal=True)
        drug = items[names.index(pick)]
        try:
            permit = search_permit(drug["itemName"])
        except Exception:
            permit = None
        drug_text = drug_to_text(drug, permit)

        if permit is None:
            st.caption("⚠️ 제품허가정보(주성분) 조회 실패 — e약은요 정보만으로 생성합니다. API 활용신청·엔드포인트를 확인하세요.")
        if permit and "전문" in str(permit.get("ETC_OTC_CODE", "")):
            st.error("전문의약품입니다. 처방전 없이 판매할 수 없어요.")

        with st.expander("식약처 원문 보기"):
            st.text(drug_text)

        with st.spinner(f"{language} 안내 + 발음 카드 생성 중…"):
            try:
                out = generate(drug_text, language)
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
            st.caption("근거: 식약처 e약은요 · AI 생성 안내, 약사 확인 후 제공")

        with right:
            st.subheader("🗣️ 약사가 직접 말하기")
            for p in out.get("key_phrases", []):
                with st.container(border=True):
                    c1, c2 = st.columns([5, 1])
                    c1.markdown(f"**{p.get('ko','')}**")
                    c1.markdown(f"<span style='font-size:1.6em'>{p.get('native','')}</span>", unsafe_allow_html=True)
                    c1.markdown(f"`{p.get('roman','')}` · **{p.get('hangul','')}**")
                    c1.caption(f"💡 {p.get('tip','')}")
                    with c2:
                        tts_button(p.get("native", ""), tts_code)
