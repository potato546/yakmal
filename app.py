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

MODEL = "gemini-3.1-flash-lite"  # 무료 한도가 넉넉한 모델. 결제 연결 후엔 gemini-3.6-flash 로 바꿔도 됨
DATA_KEY = unquote(DATA_GO_KR_KEY)

# 식약처 API (버전 번호가 바뀌면 이 줄들만 수정)
PERMIT_URL = "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService07/getDrugPrdtPrmsnDtlInq06"  # 의약품 허가정보
EASY_DRUG_URL = "https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"        # e약은요 (보조)
HTFS_URL = "https://apis.data.go.kr/1471000/HtfsInfoService03/getHtfsItem01"                        # 건강기능식품
COSM_URL = "https://apis.data.go.kr/1471000/FtnltCosmRptPrdlstInfoService01/getRptPrdlstInq01"        # 기능성화장품 보고품목

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
    for v in search_variants(q):
        items = _get(COSM_URL, {"item_name": v, "numOfRows": rows})
        items = [it for it in items if it.get("CANCEL_APPROVAL_YN") != "Y"]  # 취하된 보고 제외
        if items:
            return [{"kind": "기능성화장품", "name": it.get("ITEM_NAME", ""), "maker": it.get("ENTP_NAME", ""), "raw": it} for it in items]
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


def build_cosm_text(c: dict, ingredients: str = "") -> str:
    lines = [
        "[제품군] 기능성화장품 (의약품 아님 — 치료·개선 단정 표현 금지. 식약처에 보고된 기능성 문구 '~에 도움을 준다'만 사용. "
        "'바른다/사용한다' 표현, '복용' 금지)",
        f"[제품명] {clean(c.get('ITEM_NAME'))}",
        f"[책임판매업체] {clean(c.get('ENTP_NAME'))}",
        f"[식약처 보고 기능성] {clean(c.get('EE_NAME'))}",
    ]
    if c.get("SPF") or c.get("PA"):
        lines.append(f"[자외선차단] SPF {c.get('SPF') or '-'} / PA {c.get('PA') or '-'}")
    if c.get("USAGE_DOSAGE"):
        lines.append(f"[사용법(보고 내용)] {clean(c.get('USAGE_DOSAGE'))}")
    if c.get("REPORT_DATE"):
        lines.append(f"[보고일] {c['REPORT_DATE']}")
    if ingredients.strip():
        lines.append(f"[전성분(포장 표시, 약사가 입력)] {clean(ingredients)[:1500]}")
    else:
        lines.append("[전성분] 제공되지 않음 — 성분 설명은 하지 말고 ingredients는 빈 배열")
    return "\n".join(lines)


# ---------- Gemini ----------
SYSTEM = """당신은 한국 약국에서 외국인 고객을 응대하는 약사를 돕는 도우미입니다.
반드시 아래 JSON 하나만 반환하세요. JSON 외의 텍스트, 마크다운 펜스는 금지.
원문 첫 줄의 [제품군]을 반드시 확인하세요.
- 의약품: 효능효과 원문의 동사(개선/완화/치료/예방)를 그대로 따르고, 원문이 "개선"이면 "치료"로 올리지 말 것.
- 건강기능식품: 절대 "치료", "예방", "효과가 있다"라고 하지 말고, 식약처 인정 기능성 문구대로 "~에 도움을 줄 수 있어요"만 사용.
  "복용" 대신 "섭취", "약" 대신 "건강기능식품/영양제"라고 부를 것.
- 기능성화장품: 식약처 보고 기능성(미백/주름개선/자외선차단 등)을 "~에 도움을 줘요"로만. 치료·질환 표현 금지.
  how_to_take는 사용법(바르는 법·순서·양), cautions는 화장품 일반 주의(눈에 들어가지 않게, 이상 시 사용 중지, 자외선차단제는 덧바르기).
  전성분이 주어졌을 때만 ingredients에 주요 성분 3~5개(미백·주름·보습·진정 등 핵심 역할 성분 위주)를 넣고, 역할 설명은 일반 화장품 성분 지식으로 보수적으로.
  key_phrases (a)는 "미백에 도움을 줘요", "주름 개선에 도움을 줘요", "SPF 50이에요" 같은 것, (b)는 사용법(아침저녁, 마지막 단계에, 덧바르세요), (c)는 확인(민감성 피부세요?, 눈 주변 피하세요).
원문에는 [허가사항] 전문과, 있을 경우 [쉬운설명]이 함께 옵니다. 내용은 허가사항이 기준이고, 쉬운설명은 표현 참고용.
주의사항 전문은 매우 길 수 있으니 일반 고객에게 실제로 중요한 것(금기, 흔한 부작용, 병용 주의)만 추리세요.

1. patient_guide: 고객에게 보여줄 안내. 대상 언어로, 짧은 문장, 쉬운 표현.
   각 항목은 {"ko": 한국어, "tr": 대상 언어} 쌍. ko는 약사가 번역을 검토하기 위한 것이므로 tr과 내용이 정확히 일치해야 함.
   - what_it_is: 2~3문장. 첫 문장은 효능(또는 기능성)을 쉬운 말로. 이어서 주성분이 어떻게 작용하는지 한 문장 (일반 약학 지식, 보수적으로).
   - how_to_take, cautions, see_pharmacist_if: 원문에 있는 내용만 근거로. 없는 정보는 지어내지 마세요.
   - ingredients: 주성분·원료 정보가 있을 때만. 성분별 배열 [{"name_ko":..., "name_tr":..., "amount":..., "role_ko":..., "role_tr":...}].
     name은 고객이 알아들을 이름 (예: "빌베리건조엑스" → "빌베리 추출물(안토시아닌)"). role은 1~2문장, 과장 없이.
     같은 계열이 여러 개면 묶어도 됨. 정보가 없으면 빈 배열. 반드시 객체 배열로.

2. key_phrases: 한국인 약사가 이 제품을 팔면서 직접 입으로 말하면 효과적인 단어·짧은 구 8~9개.
   반드시 아래 세 묶음을 순서대로 포함:
   (a) 효능 3~4개: 이 제품 고유의 것. "대상 + 동사" 짧은 문장 (의약품: "야맹증을 개선해요" / 건기식: "면역력에 도움을 줄 수 있어요").
       성분 1개 포함 (예: "홍삼 성분이에요").
   (b) 복용 2~3개: 실제 용법·섭취방법에서 (예: 하루 두세 번, 한 캡슐, 식후, 물에 타서).
   (c) 확인 2개: 약사가 물어보거나 알려줄 것 (예: 당뇨약 드세요?, 2주 지나도 안 나으면 병원).
   각 항목에 "group": "효능" | "복용" | "확인" 을 넣을 것.
   각 항목: {"group":..., "ko": 한국어, "native": 대상 언어 표기, "roman": 로마자(성조 부호 포함, 라틴문자 언어는 원문 그대로),
             "hangul": 한국인이 읽기 쉬운 한글 근사 발음, "tip": 성조/강세/발음 팁 한 줄}
   광둥어는 native를 번체자 홍콩 구어체로, roman은 Jyutping(예: gam2 mou6 joek6)으로.
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
            raw = raw[raw.find("{"):]  # 앞의 잡글 제거
            obj, _ = json.JSONDecoder().raw_decode(raw)  # 첫 JSON만 읽고 뒤는 무시
            return obj
        except Exception as e:
            last_err = e
            if "503" in str(e) or ("429" in str(e) and "per_day" not in str(e).lower()):
                time.sleep(3)
                continue
            raise
    raise last_err


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


# ---------- 화면 ----------
st.set_page_config(page_title="약말", page_icon="💊", layout="wide")
st.title("💊 약말 — 외국인 복약안내 + 약사 발음 도우미")

if not DATA_GO_KR_KEY or not GEMINI_API_KEY:
    st.warning("Secrets에 DATA_GO_KR_KEY, GEMINI_API_KEY를 넣어주세요.")
    st.stop()

col_q, col_l = st.columns([3, 1])
query = col_q.text_input("제품명 검색 (예: 타이레놀, 우황청심원, 홍삼정, 리쥬올)")
language = col_l.selectbox("고객 언어", list(LANGUAGES))

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
            st.info(f"기능성화장품 — 식약처 보고 기능성: {clean(raw.get('EE_NAME'))}")
            ingr_text = st.text_area("전성분 붙여넣기 (선택) — 포장의 전성분표를 입력하면 주요 성분 설명이 추가됩니다", height=90)
            source_text = build_cosm_text(raw, ingr_text)
            basis = "식약처 기능성화장품 보고품목정보" + (" + 포장 전성분" if ingr_text.strip() else "")

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
            phrases = [p for p in (out.get("key_phrases") or []) if isinstance(p, dict)]
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
