import gzip
import io
import json
import os
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
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")          # 선택: 우리 약국 목록을 앱에서 바로 깃허브에 저장
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "")            # 예: potato546/yakmal
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
DUR_BASE = "https://apis.data.go.kr/1471000/DURPrdlstInfoService03/"  # DUR 품목정보
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
    "힌디어": ("hi-IN", "대기음(유기음)과 무기음 구분, 권설음"),
    "벵골어": ("bn-BD", "대기음 구분, 모음 조화"),
    "네팔어": ("ne-NP", "대기음 구분, 힌디어와 유사한 자음 체계"),
    "우즈베크어": ("uz-UZ", "모음 조화 약함, 라틴 문자 표기 기준"),
    "싱할라어": ("si-LK", "장단모음 구분, 비음화 자음"),
    "미얀마어": ("my-MM", "성조 없음, 창법(創法) 자음군, 성문 파열음"),
    "크메르어(캄보디아)": ("km-KH", "성조 없음, 모음 종류가 매우 많음(장단+음질)"),
}

# 언어 → 국가·지역 맥락 (반입 규제·종교·식이 안내용)
COUNTRY = {
    "영어": "영어권 관광객 (미국·영국·호주·캐나다·싱가포르·필리핀 등 — 국가를 확정할 수 없으니 주요국 공통 사항 위주)",
    "중국어(보통화)": "중국 본토", "중국어(번체·대만)": "대만", "광둥어": "홍콩·마카오", "일본어": "일본",
    "베트남어": "베트남", "태국어": "태국", "인도네시아어": "인도네시아 (무슬림 다수, 할랄 관심 높음)",
    "아랍어": "중동 아랍권 (UAE·사우디·카타르 등, 무슬림 다수, 할랄·알코올 민감)", "터키어": "터키 (무슬림 다수)",
    "몽골어": "몽골", "프랑스어": "프랑스·프랑스어권", "독일어": "독일·오스트리아·스위스", "이탈리아어": "이탈리아",
    "러시아어": "러시아·CIS", "스페인어": "스페인·중남미",
    "힌디어": "인도", "벵골어": "방글라데시·인도 서벵골", "네팔어": "네팔",
    "우즈베크어": "우즈베키스탄", "싱할라어": "스리랑카", "미얀마어": "미얀마", "크메르어(캄보디아)": "캄보디아",
}
FLAGS = ["할랄(무슬림)", "비건·채식", "알코올 금기", "글루텐 주의", "유당 불내증"]

# gTTS(구글 음성) 언어 코드. 없는 언어(광둥어·몽골어)는 브라우저 음성으로 폴백
GTTS_LANG = {
    "영어": "en", "중국어(보통화)": "zh-CN", "중국어(번체·대만)": "zh-TW", "일본어": "ja", "베트남어": "vi",
    "태국어": "th", "인도네시아어": "id", "아랍어": "ar", "터키어": "tr", "프랑스어": "fr", "독일어": "de",
    "이탈리아어": "it", "러시아어": "ru", "스페인어": "es",
    "힌디어": "hi", "벵골어": "bn", "네팔어": "ne",
    # 우즈베크어·싱할라어·미얀마어·크메르어는 gTTS 미지원 — 브라우저 음성으로 폴백
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


# ---------- 건강기능식품 로컬 데이터셋 (식품안전나라 C003, 09~19시 API 제한 우회) ----------
HTFS_LOCAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "htfs_c003.jsonl.gz")


@st.cache_resource(show_spinner=False)
def load_htfs_local() -> list[dict]:
    """식품안전나라 건강기능식품 품목제조신고(C003) 로컬 사본. 없으면 빈 목록 (앱은 정상 동작)."""
    if not os.path.exists(HTFS_LOCAL_FILE):
        return []
    out = []
    with gzip.open(HTFS_LOCAL_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def search_htfs_local(q: str, rows: int = 8) -> list[dict]:
    data = load_htfs_local()
    if not data:
        return []
    key = q.replace(" ", "")
    hits = [d for d in data if key in (d.get("PRDLST_NM") or "").replace(" ", "")]
    out = []
    for d in hits[:rows]:
        it = {"PRDUCT": d.get("PRDLST_NM", ""), "ENTRPS": d.get("BSSH_NM", ""),
              "SRV_USE": d.get("NTK_MTHD", ""), "MAIN_FNCTN": d.get("PRIMARY_FNCLTY", ""),
              "INTAKE_HINT1": d.get("IFTKN_ATNT_MATR_CN", ""), "SUNGSANG": d.get("PRDT_SHAP_CD_NM", ""),
              "STTEMNT_NO": d.get("PRDLST_REPORT_NO", ""), "RAWMTRL_NM": d.get("RAWMTRL_NM", "")}
        out.append({"kind": "건강기능식품", "name": it["PRDUCT"].strip(), "maker": it["ENTRPS"], "raw": it})
    return out



@st.cache_data(ttl=3600, show_spinner=False)
def search_htfs(q: str, rows: int = 6) -> list[dict]:
    for v in search_variants(q):
        try:
            items = _get(HTFS_URL, {"Prduct": v, "numOfRows": rows})
        except Exception:
            items = []
        if items:
            return [{"kind": "건강기능식품", "name": it.get("PRDUCT", "").strip(), "maker": it.get("ENTRPS", ""), "raw": it} for it in items]
        local = search_htfs_local(v, rows)
        if local:
            return local
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
    chart = clean(permit.get("CHART"))
    if chart:
        lines.append(f"[제형·성상] {chart}")
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
                       ("보관방법", "PRSRV_PD"), ("유통기한", "DISTB_PD"), ("기준규격", "BASE_STANDARD"),
                       ("원재료명", "RAWMTRL_NM")]:
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


NAVER_INGR_SYSTEM = """네이버 검색 결과와 페이지 본문에서 특정 화장품의 정보를 뽑습니다.
반드시 해당 제품(제품명이 일치)의 것만 사용. 다른 제품 것은 섞지 말 것.
JSON만 반환 (마크다운 펜스 금지):
{"found": true/false, "partial": true/false,
 "ingredients": "전성분(또는 확인된 주요 성분)을 표시 순서대로 쉼표로 이어 쓴 문자열",
 "disclosed_amounts": "회사가 공개한 함량 (예: PDRN 2%), 없으면 빈 문자열",
 "texture": "제형·사용감 (예: 젤 크림, 가벼움, 끈적임 없음) 한 줄. 광고 표현 제외",
 "source": "출처 페이지 제목 또는 URL", "note": "확인 범위·주의 한 줄"}"""


@st.cache_data(ttl=86400, show_spinner=False)
def naver_ingredients(name: str, maker: str) -> dict:
    hits = []
    for q in (f"{name} 전성분", f"{name} 성분"):
        for kind in ("webkr", "blog"):
            try:
                hits += naver_search(kind, q, display=10)
            except Exception:
                pass
    seen, pages = set(), []
    for h in hits:
        link = h.get("link", "")
        if not link or link in seen:
            continue
        seen.add(link)
        body = fetch_text(link, 5000)
        if "전성분" in body or "성분" in body:
            pages.append(f"### {clean(strip_html(h.get('title','')))} ({link})\n{body}")
        if len(pages) >= 5:
            break
    if not pages:
        return {"found": False, "note": "검색 결과에서 성분 텍스트를 찾지 못함"}
    ctx = f"제품명: {name} / 판매사: {maker}\n\n" + "\n\n".join(pages)
    resp = client.models.generate_content(
        model=MODEL, contents=ctx[:30000],
        config=types.GenerateContentConfig(system_instruction=NAVER_INGR_SYSTEM, response_mime_type="application/json",
                                           thinking_config=types.ThinkingConfig(thinking_level="low")),
    )
    raw = re.sub(r"^```(?:json)?|```$", "", (resp.text or "").strip(), flags=re.M).strip()
    raw = raw[raw.find("{"):]
    obj, _ = json.JSONDecoder().raw_decode(raw)
    return obj


REVIEW_SYSTEM = """약사가 참고할 실사용 후기 요약을 만듭니다. 입력은 네이버 블로그·카페 글 목록(제목·요약)입니다.
협찬·광고·판매 목적 글("협찬", "제공받아", "광고", "공동구매", 지나친 찬사, 구매 링크 유도)은 제외하고 개인 경험담만 사용.
고객에게 보여주는 글이 아니라 약사 내부 참고용. 효능 근거로 쓰지 말 것을 전제로, 담백하게.
JSON만 반환 (마크다운 펜스 금지):
{"used": 사용한 글 수, "excluded_ads": 제외한 협찬 의심 글 수,
 "effects": ["체감 효과로 자주 언급된 것 (몇 건인지 괄호로)", ...],
 "side_effects": ["부작용·불편으로 언급된 것 (건수)", ...],
 "tips": ["복용·사용 팁, 사용감, 제형 등", ...],
 "caution": "약사가 주의해서 볼 점 한 줄 (예: 졸림 언급이 많음)"}"""


@st.cache_data(ttl=86400, show_spinner=False)
def naver_reviews(name: str, kind_label: str) -> dict:
    hits = []
    for q in (f"{name} 후기", f"{name} 부작용" if kind_label != "기능성화장품" else f"{name} 사용감"):
        for k in ("blog", "cafearticle"):
            try:
                hits += naver_search(k, q, display=20)
            except Exception:
                pass
    seen, lines = set(), []
    for h in hits:
        link = h.get("link", "")
        if link in seen:
            continue
        seen.add(link)
        lines.append(f"- [{clean(strip_html(h.get('title','')))}] {clean(strip_html(h.get('description','')))} ({h.get('postdate') or h.get('pubDate','')})")
    if not lines:
        return {"used": 0, "excluded_ads": 0, "effects": [], "side_effects": [], "tips": [], "caution": "후기를 찾지 못함"}
    ctx = f"제품: {name} ({kind_label})\n\n" + "\n".join(lines[:60])
    resp = client.models.generate_content(
        model=MODEL, contents=ctx,
        config=types.GenerateContentConfig(system_instruction=REVIEW_SYSTEM, response_mime_type="application/json",
                                           thinking_config=types.ThinkingConfig(thinking_level="low")),
    )
    raw = re.sub(r"^```(?:json)?|```$", "", (resp.text or "").strip(), flags=re.M).strip()
    raw = raw[raw.find("{"):]
    obj, _ = json.JSONDecoder().raw_decode(raw)
    return obj


def render_reviews(name: str, kind_label: str):
    if not naver_ok():
        return
    with st.expander("📝 약사 참고: 실사용 후기 요약 (네이버 블로그·카페) — 고객 안내에는 사용되지 않음"):
        if st.button("후기 모아 보기", key=f"rv_{name}"):
            with st.spinner("후기 수집·정리 중…"):
                try:
                    R = naver_reviews(name, kind_label)
                except Exception as e:
                    st.error(f"후기 조회 실패: {str(e)[:200]}")
                    return
            st.caption(f"개인 경험담 {R.get('used',0)}건 요약 · 협찬 의심 {R.get('excluded_ads',0)}건 제외 · 효능·안전성 근거 아님")
            for label, k in [("체감 효과", "effects"), ("부작용·불편", "side_effects"), ("팁·사용감", "tips")]:
                items = [x for x in (R.get(k) or []) if isinstance(x, str)]
                if items:
                    st.markdown(f"**{label}**")
                    for x in items:
                        st.markdown(f"- {x}")
            if R.get("caution"):
                st.warning(R["caution"])


# ---------- DUR 판매 전 확인 ----------
DUR_OPS = {
    "임부금기": "getPwnmTabooInfoList03",
    "특정연령대금기": "getSpcifyAgrdeTabooInfoList03",
    "노인주의": "getOdsnAtentInfoList03",
    "용량주의": "getCpctyAtentInfoList03",
    "투여기간주의": "getMdctnPdAtentInfoList03",
    "병용금기": "getUsjntTabooInfoList03",
    "효능군중복": "getEfcyDplctInfoList03",
    "서방정분할주의": "getSeobangjeongPartitnAtentInfoList03",
}


@st.cache_data(ttl=86400, show_spinner=False)
def dur_lookup(op: str, item_seq: str, rows: int = 50) -> list[dict]:
    try:
        items, _ = _get(DUR_BASE + op, {"itemSeq": item_seq, "numOfRows": rows})
        return items
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def dur_all(item_seq: str) -> dict[str, list[dict]]:
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {k: ex.submit(dur_lookup, op, item_seq) for k, op in DUR_OPS.items()}
        return {k: f.result() for k, f in futs.items()}


def dur_text(it: dict) -> str:
    return clean(it.get("PROHBT_CONTENT") or it.get("REMARK") or "")


def ingr_names_of(permit: dict) -> list[str]:
    names = []
    for chunk in (permit.get("MATERIAL_NAME") or "").split(";"):
        for part in chunk.split("|"):
            if "성분명" in part and ":" in part:
                names.append(part.split(":", 1)[1].strip())
    return [n for n in names if n]


@st.cache_data(ttl=86400, show_spinner=False)
def resolve_other_drug(q: str) -> tuple[str, list[str]]:
    """복용 중인 약 입력(제품명 또는 성분명) → (표시명, 성분명 목록)"""
    q = q.strip()
    if not q:
        return "", []
    try:
        res = search_drug(q, rows=3)
    except Exception:
        res = []
    if res:
        p = res[0]["raw"]
        return res[0]["name"], ingr_names_of(p) or [q]
    return q, [q]


def _norm(x: str) -> str:
    return re.sub(r"[\s\(\)\[\]·,.\-]", "", x or "").lower()


def interaction_hits(dur: dict, other_ingrs: list[str], other_name: str) -> list[tuple[str, dict]]:
    """병용금기·효능군중복 목록에서 상대 약 성분/제품명이 걸리는 항목"""
    keys = [_norm(x) for x in other_ingrs + [other_name] if x]
    out = []
    for cat in ("병용금기", "효능군중복"):
        for it in dur.get(cat, []):
            hay = _norm(" ".join(str(it.get(k) or "") for k in ("MIXTURE_INGR_KOR_NAME", "MIXTURE_ITEM_NAME", "MIXTURE_INGR_ENG_NAME")))
            if any(k and (k in hay or hay and hay in k) for k in keys):
                out.append((cat, it))
    return out


CHECK_Q_SYSTEM = """약국에서 약을 팔기 전에 외국인 고객에게 물어볼 확인 질문을 대상 언어로 만듭니다. 짧고 예의 바르게, 한 문장씩.
JSON만 반환: {"pregnant": {"ko":"임신 중이거나 수유 중이세요?","tr":""},
             "age": {"ko":"본인이 드실 건가요? 아이나 어르신이 드시나요?","tr":""},
             "other": {"ko":"지금 드시는 약이 있으세요?","tr":""},
             "allergy": {"ko":"약 알레르기가 있으세요?","tr":""}}"""


@st.cache_data(ttl=86400, show_spinner=False)
def check_questions(language: str) -> dict:
    try:
        return _gen_json(CHECK_Q_SYSTEM, f"대상 언어: {language}")
    except Exception:
        return {}


def render_dur(permit: dict, language: str):
    seq = permit.get("ITEM_SEQ", "")
    if not seq:
        return
    dur = dur_all(seq)
    qs = check_questions(language)

    st.subheader("🛡️ 판매 전 확인 (식약처 DUR)")
    found = {k: v for k, v in dur.items() if v and k not in ("병용금기", "효능군중복")}
    n_inter = len(dur.get("병용금기", [])) + len(dur.get("효능군중복", []))
    if not found and not n_inter:
        st.caption("이 품목은 DUR 등록 항목이 없습니다. (금기·주의 정보 없음)")
    else:
        st.caption("등록 항목: " + " · ".join([f"{k} {len(v)}" for k, v in found.items()] + ([f"병용금기/효능군중복 {n_inter}"] if n_inter else [])))

    def q(key, default):
        tr, ko, _ = pair(qs.get(key)) if qs else ("", "", "")
        return f"{tr or ''}  —  {ko or default}".strip(" —")

    c1, c2 = st.columns(2)
    preg = c1.checkbox(q("pregnant", "임신 중이거나 수유 중이세요?"), key=f"preg_{seq}")
    age = c2.radio(q("age", "본인이 드실 건가요? 아이나 어르신이 드시나요?"), ["성인 본인", "어린이·청소년", "65세 이상"], horizontal=True, key=f"age_{seq}")
    other = st.text_input(q("other", "지금 드시는 약이 있으세요?") + "  (제품명 또는 성분명, 쉼표로 여러 개)", key=f"other_{seq}",
                          placeholder="예: 타이레놀, 아스피린, ibuprofen")

    alerts = []
    if preg and dur.get("임부금기"):
        for it in dur["임부금기"][:3]:
            alerts.append(("red", f"임부금기 {clean(it.get('GRADE') or '')}: {dur_text(it)}"))
    if age == "어린이·청소년" and dur.get("특정연령대금기"):
        for it in dur["특정연령대금기"][:3]:
            alerts.append(("red", f"연령금기: {dur_text(it)}"))
    if age == "65세 이상" and dur.get("노인주의"):
        for it in dur["노인주의"][:3]:
            alerts.append(("yellow", f"노인주의: {dur_text(it)}"))
    for it in dur.get("용량주의", [])[:2]:
        alerts.append(("yellow", f"용량주의: {dur_text(it)}"))
    for it in dur.get("투여기간주의", [])[:2]:
        alerts.append(("yellow", f"투여기간주의: {dur_text(it)}"))
    for it in dur.get("서방정분할주의", [])[:1]:
        alerts.append(("yellow", f"서방정 분할주의: {dur_text(it)}"))
    if other.strip():
        for one in [x for x in other.split(",") if x.strip()]:
            name, ingrs = resolve_other_drug(one)
            hits = interaction_hits(dur, ingrs, name)
            if hits:
                for cat, it in hits[:3]:
                    alerts.append(("red" if cat == "병용금기" else "yellow",
                                   f"{cat} — {name}: {clean(it.get('MIXTURE_INGR_KOR_NAME') or '')} {dur_text(it)}"))
            else:
                alerts.append(("green", f"{name}: DUR 병용금기·효능군중복 해당 없음 (성분: {', '.join(ingrs[:3])})"))

    for level, msg in alerts:
        (st.error if level == "red" else st.warning if level == "yellow" else st.success)(msg)
    if not alerts:
        st.info("체크한 조건에서 걸리는 DUR 항목이 없습니다.")
    st.caption("DUR = 식약처 의약품안전사용서비스. 최종 판단은 약사가 합니다.")


# ---------- 우리 약국 목록 (등록·저장) ----------
import base64

MY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_pharmacy.txt")


def parse_my_lines(text: str) -> list[dict]:
    """'제품군 | 제품명 | 제조사' 또는 '제품명' 한 줄씩"""
    out = []
    for line in text.splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        parts = [x.strip() for x in t.split("|")]
        if len(parts) >= 3:
            out.append({"kind": parts[0], "name": parts[1], "maker": parts[2]})
        elif len(parts) == 2:
            out.append({"kind": parts[0], "name": parts[1], "maker": ""})
        else:
            out.append({"kind": "", "name": parts[0], "maker": ""})
    return out


def my_lines_text(items: list[dict]) -> str:
    head = "# 우리 약국 제품 목록 — 앱에서 ⭐로 추가/삭제하고 저장하세요. 형식: 제품군 | 제품명 | 제조사\n"
    return head + "\n".join((i['name'] if i['kind'] in ('', '?') else f"{i['kind']} | {i['name']} | {i['maker']}") for i in items) + "\n"


def load_my_from_file() -> list[dict]:
    if not os.path.exists(MY_FILE):
        return []
    with open(MY_FILE, encoding="utf-8") as f:
        return parse_my_lines(f.read())


@st.cache_data(ttl=86400, show_spinner=False)
def resolve_my_name(name: str) -> dict | None:
    """제품명만 적힌 줄을 식약처 검색으로 확정 (이름이 정확히 같은 것 우선)"""
    try:
        results, _ = search_all(name)
    except Exception:
        return None
    if not results:
        return None
    key = name.replace(" ", "")
    for r in results:
        if r["name"].replace(" ", "") == key:
            return {"kind": r["kind"], "name": r["name"], "maker": r["maker"]}
    r = results[0]
    return {"kind": r["kind"], "name": r["name"], "maker": r["maker"]}


def resolve_my_items(items: list[dict]) -> list[dict]:
    """kind가 비어 있는 항목만 병렬로 확정. 못 찾으면 kind='?'로 남김"""
    todo = [i for i, x in enumerate(items) if not x.get("kind")]
    if not todo:
        return items
    with ThreadPoolExecutor(max_workers=6) as ex:
        found = list(ex.map(lambda i: resolve_my_name(items[i]["name"]), todo))
    out = list(items)
    for i, r in zip(todo, found):
        out[i] = r if r else {"kind": "?", "name": items[i]["name"], "maker": ""}
    return out


def my_key(r: dict) -> tuple:
    return (r.get("kind", ""), r.get("name", "").replace(" ", ""))


def my_add(r: dict):
    items = st.session_state["my_products"]
    if my_key(r) not in {my_key(x) for x in items}:
        items.append({"kind": r["kind"], "name": r["name"], "maker": r.get("maker", "")})
        st.session_state["my_dirty"] = True


def my_remove(key: tuple):
    st.session_state["my_products"] = [x for x in st.session_state["my_products"] if my_key(x) != key]
    st.session_state["my_dirty"] = True


def github_save(text: str) -> str:
    """my_pharmacy.txt를 깃허브에 커밋. 성공 시 빈 문자열, 실패 시 오류 메시지."""
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return "GITHUB_TOKEN / GITHUB_REPO 가 Secrets에 없음"
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/my_pharmacy.txt"
    h = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    try:
        cur = requests.get(url, headers=h, timeout=10)
        sha = cur.json().get("sha") if cur.status_code == 200 else None
        body = {"message": "우리 약국 목록 갱신 (앱)", "content": base64.b64encode(text.encode("utf-8")).decode()}
        if sha:
            body["sha"] = sha
        r = requests.put(url, headers=h, json=body, timeout=15)
        return "" if r.status_code in (200, 201) else f"{r.status_code} {r.text[:120]}"
    except Exception as e:
        return str(e)[:150]


def render_my_sidebar():
    with st.sidebar:
        st.markdown("### 🏪 우리 약국 목록")
        items = st.session_state["my_products"]
        if not items:
            st.caption("검색 결과에서 ⭐ 버튼으로 추가하세요.")
        for x in items:
            c1, c2 = st.columns([5, 1])
            cls = 'drug' if x['kind'] == '의약품' else 'htfs' if x['kind'] == '건강기능식품' else 'cosm'
            if x['kind'] == '?':
                c1.markdown(f"❓ {esc(x['name'])} <span class='ym-basis'>(식약처에서 못 찾음 — 이름 수정)</span>", unsafe_allow_html=True)
            else:
                c1.markdown(f"<span class='ym-tag {cls}'>{esc(x['kind'][:3])}</span> {esc(x['name'])}", unsafe_allow_html=True)
            if c2.button("✕", key=f"rm_{my_key(x)}"):
                my_remove(my_key(x))
                st.rerun()
        st.divider()
        txt = my_lines_text(items)
        if st.session_state.get("my_dirty"):
            st.caption("변경됨 — 저장하지 않으면 앱 재시작 시 사라져요.")
        if GITHUB_TOKEN and GITHUB_REPO:
            if st.button("💾 깃허브에 저장", use_container_width=True):
                err = github_save(txt)
                if err:
                    st.error(f"저장 실패: {err}")
                else:
                    st.session_state["my_dirty"] = False
                    st.success("저장됨. 1분 뒤 앱이 새 목록으로 재시작해요.")
        st.download_button("⬇️ my_pharmacy.txt 내려받기", txt, file_name="my_pharmacy.txt", use_container_width=True)
        up = st.file_uploader("목록 파일 불러오기", type=["txt"], label_visibility="collapsed")
        if up is not None:
            st.session_state["my_products"] = resolve_my_items(parse_my_lines(up.getvalue().decode("utf-8", "ignore")))
            st.session_state["my_dirty"] = True
            st.rerun()
        if not (GITHUB_TOKEN and GITHUB_REPO):
            st.caption("영구 저장: 내려받은 파일을 깃허브의 my_pharmacy.txt에 붙여넣기. (Secrets에 GITHUB_TOKEN·GITHUB_REPO를 넣으면 버튼 한 번으로 저장돼요)")

import os

INGR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cosmetic_ingredients.txt")


def load_ingredient_file() -> dict[str, str]:
    """cosmetic_ingredients.txt: '제품명 | 전성분' 한 줄씩. 제품명은 띄어쓰기 무시하고 부분일치."""
    table = {}
    if not os.path.exists(INGR_FILE):
        return table
    with open(INGR_FILE, encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#") or "|" not in t:
                continue
            name, ingr = t.split("|", 1)
            table[name.strip().replace(" ", "")] = ingr.strip()
    return table


def file_ingredients(product_name: str) -> str:
    key = product_name.replace(" ", "")
    for name, ingr in load_ingredient_file().items():
        if name in key or key in name:
            return ingr
    return ""


PHOTO_SYSTEM = """화장품 포장 사진에서 전성분 표기를 읽어 정리합니다.
'전성분' 또는 'Ingredients' 뒤에 나오는 성분 목록을 표시된 순서 그대로, 쉼표로 구분해 한 줄로 옮기세요.
사진에 보이는 그대로만 적고 추측해서 채우지 마세요. 흐려서 못 읽는 부분은 (판독불가)로 표시.
회사가 표시한 함량(예: 나이아신아마이드 2%)이 보이면 그대로 포함.
JSON만 반환 (마크다운 펜스 금지):
{"found": true/false, "product_name": "사진에서 보이는 제품명(없으면 빈 문자열)",
 "ingredients": "성분1, 성분2, ...", "note": "판독 상태 한 줄"}"""


def read_ingredients_from_photo(img_bytes: bytes, mime: str) -> dict:
    resp = client.models.generate_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=img_bytes, mime_type=mime), "이 사진의 전성분을 읽어 JSON으로."],
        config=types.GenerateContentConfig(
            system_instruction=PHOTO_SYSTEM,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    raw = re.sub(r"^```(?:json)?|```$", "", (resp.text or "").strip(), flags=re.M).strip()
    raw = raw[raw.find("{"):]
    obj, _ = json.JSONDecoder().raw_decode(raw)
    return obj


def build_cosm_text(c: dict, ingredients: str = "") -> str:
    lines = [
        "[제품군] 기능성화장품 (의약품 아님 — 치료·개선 단정 표현 금지. 식약처에 보고된 기능성 문구 '~에 도움을 준다'만 사용. "
        "'바른다/사용한다' 표현, '복용' 금지)",
        f"[제품명] {clean(c.get('ITEM_NAME'))}",
        f"[책임판매업체] {clean(c.get('ENTP_NAME'))}",
        f"[식약처 {c.get('_src','보고')} 기능성] {clean(c.get('EE_NAME')) or doc_text(c.get('EE_DOC_DATA')) or '기능성화장품(세부 기능성 문구 없음)'}",
    ]
    ud = doc_text(c.get("UD_DOC_DATA"))
    if ud and not c.get("USAGE_DOSAGE"):
        lines.append(f"[사용법(심사 문서)] {ud[:600]}")
    if c.get("SPF") or c.get("PA"):
        lines.append(f"[자외선차단] SPF {c.get('SPF') or '-'} / PA {c.get('PA') or '-'}")
    if c.get("USAGE_DOSAGE"):
        lines.append(f"[사용법(보고 내용)] {clean(c.get('USAGE_DOSAGE'))}")
    if c.get("REPORT_DATE"):
        lines.append(f"[보고일] {c['REPORT_DATE']}")
    if ingredients.strip():
        lines.append(f"[전성분] {clean(ingredients)[:1500]}")
    else:
        lines.append("[전성분] 제공되지 않음 — 성분 설명은 하지 말고 ingredients는 빈 배열")
    return "\n".join(lines)


def build_general_cosm_text(name: str, maker: str, ingredients: str, claims: str) -> str:
    lines = [
        "[제품군] 일반화장품 (식약처 기능성 인증 없음 — 효능 단정 금지. 제조사 자체 시험 결과는 출처를 밝혀서만 전달. 광고 형용사 배제)",
        f"[제품명] {clean(name)}",
        f"[제조사] {clean(maker)}",
        "[제형 힌트] 제품명에서 제형을 판단하세요 (토너/스킨→화장수, 크림/로션→크림, 에센스/세럼→에센스, 폼/클렌저→클렌징폼, 미스트, 팩 등). "
        "발음 카드 첫 항목은 반드시 이 제형 단어 하나.",
    ]
    if ingredients.strip():
        lines.append(f"[전성분] {clean(ingredients)[:1500]}")
    else:
        lines.append("[전성분] 제공되지 않음 — 성분 설명은 하지 말고 ingredients는 빈 배열")
    if claims.strip():
        lines.append(f"[제조사 자체 시험·표시 내용(웹 출처)] {clean(claims)[:1000]}")
    return "\n".join(lines)


NAVER_CLAIMS_SYSTEM = """네이버 검색 결과와 페이지 본문에서 화장품의 '제조사 자체 시험 결과'만 뽑습니다.
조건: 구체적 수치 + 시험 방법·기간 + (가능하면) 시험 기관명이 함께 있는 것만. "즉각 개선", "최고" 같은 형용사만 있는 문장은 버림.
반드시 해당 제품 것만 사용, 다른 제품 것 섞지 말 것.
JSON만 반환 (마크다운 펜스 금지):
{"found": true/false, "claims": "구체적 시험 결과 요약 한두 문장 (예: 12주 사용 후 피부 요철 개선율 87%, OO피부과 임상 평가)",
 "ingredients": "전성분을 찾았으면 표시 순서대로, 없으면 빈 문자열",
 "source": "출처 페이지 제목 또는 URL", "note": "확인 범위·주의 한 줄"}"""


@st.cache_data(ttl=86400, show_spinner=False)
def naver_claims(name: str, maker: str) -> dict:
    hits = []
    for q in (f"{name} 임상", f"{name} 시험 결과", f"{name} 전성분"):
        for kind in ("webkr", "blog"):
            try:
                hits += naver_search(kind, q, display=10)
            except Exception:
                pass
    seen, pages = set(), []
    for h in hits:
        link = h.get("link", "")
        if not link or link in seen:
            continue
        seen.add(link)
        body = fetch_text(link, 5000)
        if body:
            pages.append(f"### {clean(strip_html(h.get('title','')))} ({link})\n{body}")
        if len(pages) >= 5:
            break
    if not pages:
        return {"found": False, "note": "검색 결과 없음"}
    ctx = f"제품명: {name} / 제조사: {maker}\n\n" + "\n\n".join(pages)
    resp = client.models.generate_content(
        model=MODEL, contents=ctx[:30000],
        config=types.GenerateContentConfig(system_instruction=NAVER_CLAIMS_SYSTEM, response_mime_type="application/json",
                                           thinking_config=types.ThinkingConfig(thinking_level="low")),
    )
    raw = re.sub(r"^```(?:json)?|```$", "", (resp.text or "").strip(), flags=re.M).strip()
    raw = raw[raw.find("{"):]
    obj, _ = json.JSONDecoder().raw_decode(raw)
    return obj


# ---------- Gemini (안내문 + 발음 카드) ----------
COMMON = """당신은 한국 약국에서 외국인 고객을 응대하는 약사를 돕는 도우미입니다.
반드시 JSON 하나만 반환하세요. JSON 외의 텍스트, 마크다운 펜스는 금지.
원문 첫 줄의 [제품군]을 반드시 확인하세요.
- 의약품: 효능효과 원문의 동사(개선/완화/치료/예방)를 그대로 따르고, 원문이 "개선"이면 "치료"로 올리지 말 것.
- 건강기능식품: 절대 "치료", "예방", "효과가 있다"라고 하지 말고, 식약처 인정 기능성 문구대로 "~에 도움을 줄 수 있어요"만 사용.
  "복용" 대신 "섭취", "약" 대신 "건강기능식품/영양제"라고 부를 것.
- 기능성화장품: 식약처 보고 기능성(미백/주름개선/자외선차단 등)을 "~에 도움을 줘요"로만. 치료·질환 표현 금지. '복용' 금지, '바르다/사용하다'.
- 일반화장품(식약처 미등록): 식약처 인증 문구가 없으므로 효능을 단정하지 말 것. 제조사 자체 시험 결과가 구체적 수치·기관명과 함께 주어졌으면
  "제조사 자체 시험에서는 ~로 나타났어요"처럼 출처를 밝혀 전달. 광고 형용사("최고의","즉각적인","기적의" 등)는 전달하지 말고 사실만.
원문에는 [허가사항] 전문과, 있을 경우 [쉬운설명]이 함께 옵니다. 내용은 허가사항이 기준이고, 쉬운설명은 표현 참고용.
"""

SYSTEM_GUIDE = COMMON + """
patient_guide: 고객에게 보여줄 안내. 대상 언어로, 짧은 문장, 쉬운 표현.
각 항목은 {"ko": 한국어, "tr": 대상 언어} 쌍. ko는 약사가 번역을 검토하기 위한 것이므로 tr과 내용이 정확히 일치해야 함.
- 대상 언어가 일본어면 tr의 한자마다 <ruby>한자<rt>히라가나</rt></ruby> 태그를 달 것 (예: <ruby>薬<rt>くすり</rt></ruby>を飲んでください).
  히라가나·가타카나·숫자·기호는 태그 없이 그대로. 모든 한자에 빠짐없이 달 것.
- 대상 언어가 중국어(보통화/번체·대만)나 광둥어면, tr과 별도로 pinyin_pairs 필드에 그 문장의 한자를 한 글자씩(구두점 제외) 병음과 짝지어
  배열로 추가: [{"c":"한자1글자","p":"성조 부호 포함 병음"}, ...]. 광둥어는 Jyutping. 숫자·영문·구두점은 pinyin_pairs에서 제외.
  {"ko":"", "tr":"", "pinyin_pairs":[]} 형태로.
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
(a) 제형+효능 4~5개: 반드시 첫 항목은 제형 단어 하나만 (예: "크림이에요", "정제예요", "캡슐이에요", "시럽이에요", "패치예요",
    "환이에요" — [제형·성상] 또는 제품명·제형코드에서 판단). 그다음 이 제품 고유의 효능 3~4개, "대상 + 동사" 짧은 문장
    (의약품: "야맹증을 개선해요" / 건기식: "면역력에 도움을 줄 수 있어요" / 화장품: "미백에 도움을 줘요", "SPF 50이에요").
    성분 1개 포함 (예: "홍삼 성분이에요").
(b) 복용 2~3개: 실제 용법·섭취방법·사용법에서 (예: 하루 두세 번, 한 캡슐, 식후, 아침저녁, 마지막 단계에).
(c) 확인 2개: 약사가 물어보거나 알려줄 것 (예: 당뇨약 드세요?, 2주 지나도 안 나으면 병원, 민감성 피부세요?).
각 항목에 "group": "효능" | "복용" | "확인".
각 항목: {"group":..., "ko": 한국어, "native": 대상 언어 표기, "roman": 로마자(성조 부호 포함, 라틴문자 언어는 원문 그대로),
          "hangul": 한국인이 읽기 쉬운 한글 근사 발음, "tip": 성조/강세/발음 팁 한 줄}
광둥어는 native를 번체자 홍콩 구어체로, roman은 Jyutping(예: gam2 mou6 joek6)으로.
hangul은 한국어 음운으로 최대한 가깝게. tip은 한국어 화자가 틀리기 쉬운 지점을 구체적으로.
형식: {"key_phrases": []}"""


SYSTEM_COUNTRY = COMMON + """
country_notes: 이 제품을 외국인 관광객이 사서 본국으로 가져가거나 사용할 때 약사가 알려줄 국가·문화별 안전사항.
근거 등급을 반드시 구분:
- "확인": 식약처 원문(제형, 성분, 함량)에서 직접 확인되는 사실 (예: 연질캡슐 → 젤라틴 캡슐, 액제에 에탄올 함유)
- "참고": 일반 지식 (반입 규제, 본국 등가 제품, 종교·식이 관행). 규제는 바뀌므로 "출국 전 세관·대사관 확인" 전제.
과장·단정 금지. 해당 없으면 level을 "해당없음"으로. 각 text는 {"ko": 한국어, "tr": 대상 언어} 쌍 (약사가 대조·전달).
형식:
{"import": {"level": "주의"|"참고"|"해당없음", "basis": "확인"|"참고", "text": {"ko":"","tr":""}},   # 본국 반입 규제 (예: 일본 슈도에페드린·코데인 한도, 태국·UAE 향정 규제)
 "religion_diet": {"level": ..., "basis": ..., "text": {"ko":"","tr":""}},                          # 할랄(젤라틴·에탄올·돼지 유래), 비건, 알코올, 글루텐, 유당 등. 고객 특이사항이 있으면 그것 중심
 "equivalent": {"level": ..., "basis": "참고", "text": {"ko":"","tr":""}},                          # 본국에서 같은 성분의 흔한 제품·일반명 (예: 아세트아미노펜 → 일본 カロナール, 미국 Tylenol). 화장품·건기식은 성분 일반명 위주
 "pharmacist_checks": ["약사가 제조사에 확인하거나 고객에게 물어볼 것 (예: 캡슐 젤라틴 기원(소/돼지) 제조사 확인)"]}"""


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
def generate(source_text: str, language: str, flags: tuple[str, ...] = ()) -> dict:
    """안내문 / 발음 카드 / 국가별 안전사항을 세 요청으로 나눠 동시에 생성"""
    tts_code, phon = LANGUAGES[language]
    user = f"대상 언어: {language} (발음 특성: {phon})\n\n식약처 원문:\n{source_text}"
    user_c = (f"대상 언어: {language}\n고객 국가·지역: {COUNTRY.get(language, language)}\n"
              f"고객 특이사항: {', '.join(flags) if flags else '없음'}\n\n식약처 원문:\n{source_text}")
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_guide = ex.submit(_gen_json, SYSTEM_GUIDE, user)
        f_phr = ex.submit(_gen_json, SYSTEM_PHRASES, user)
        f_cty = ex.submit(_gen_json, SYSTEM_COUNTRY, user_c)
        guide = f_guide.result()
        phr = f_phr.result()
        try:
            cty = f_cty.result()
        except Exception as e:
            cty = {"_error": str(e)[:200]}
    return {"patient_guide": guide, "key_phrases": phr.get("key_phrases", []), "country_notes": cty}


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


def pair(v) -> tuple[str, str, list]:
    if isinstance(v, dict):
        pp = v.get("pinyin_pairs")
        pp = pp if isinstance(pp, list) else []
        return str(v.get("tr", "")), str(v.get("ko", "")), pp
    return str(v or ""), "", []


def esc(x) -> str:
    return (str(x or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_phrases(phrases: list[dict], language: str):
    phrases = [p for p in (phrases or []) if isinstance(p, dict)]
    tts_code = LANGUAGES.get(language, ("en-US", ""))[0]
    gl = GTTS_LANG.get(language)
    rtl = ' dir="rtl"' if language == "아랍어" else ""
    audios = make_all_audio([str(p.get("native", "")) for p in phrases], gl) if gl else [None] * len(phrases)
    last_group = None
    for p, audio in zip(phrases, audios):
        if p.get("group") and p.get("group") != last_group:
            st.markdown(f'<span class="ym-group">{esc(p["group"])}</span>', unsafe_allow_html=True)
            last_group = p["group"]
        st.markdown(f"""
<div class="ym-card">
  <div class="ym-ko">{esc(p.get('ko'))}</div>
  <div class="ym-native"{rtl}>{esc(p.get('native'))}</div>
  <div class="ym-roman">{esc(p.get('roman'))}</div>
  <div class="ym-hangul">{esc(p.get('hangul'))}</div>
  <div class="ym-tip">💡 {esc(p.get('tip'))}</div>
</div>""", unsafe_allow_html=True)
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


def render_country(c: dict, language: str):
    st.subheader(f"🌍 국가별 안전사항 · {COUNTRY.get(language, language).split(' (')[0]}")
    if c.get("_error"):
        st.caption(f"생성 실패: {c['_error']}")
        return
    rtl = ' dir="rtl"' if language == "아랍어" else ""
    icon = {"주의": "⚠️", "참고": "ℹ️", "해당없음": "✅"}
    rows = [("본국 반입", "import"), ("종교·식이", "religion_diet"), ("본국 등가 제품", "equivalent")]
    html = ['<div class="ym-guide">']
    shown = 0
    for label, k in rows:
        item = c.get(k) or {}
        if not isinstance(item, dict) or item.get("level") == "해당없음":
            continue
        tr, ko, _ = pair(item.get("text"))
        if not (tr or ko):
            continue
        shown += 1
        badge = f'<span class="ym-tag {"drug" if item.get("basis")=="확인" else "htfs"}">{esc(item.get("basis","참고"))}</span>'
        html.append(f"<h4>{icon.get(item.get('level'),'')} {esc(label)} {badge}</h4><p{rtl}>{esc(tr)}</p>"
                    + (f'<p class="ko">{esc(ko)}</p>' if ko else ""))
    checks = [x for x in (c.get("pharmacist_checks") or []) if isinstance(x, str)]
    if checks:
        html.append("<h4>약사 확인 사항</h4>" + "".join(f"<p>• {esc(x)}</p>" for x in checks))
    if shown == 0 and not checks:
        html.append("<p>이 제품·국가 조합에서 특별히 알려줄 사항이 없습니다.</p>")
    html.append('<div class="ym-basis">「확인」= 식약처 원문 근거 · 「참고」= 일반 지식(반입 규제는 출국 전 세관·대사관 확인)</div></div>')
    st.markdown("".join(html), unsafe_allow_html=True)


# ---------- 화면 ----------
st.set_page_config(page_title="약말 · 외국인 응대 발음 도우미", page_icon="💊", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
#MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}
.block-container {padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1100px;}
.ym-brand {display:flex; align-items:baseline; gap:.6rem; margin-bottom:.2rem;}
.ym-brand h1 {font-size: 1.9rem; margin:0; letter-spacing:-.02em; color:#123b32;}
.ym-brand span {color:#5c6b66; font-size:.95rem;}
.ym-sub {color:#5c6b66; font-size:.9rem; margin: 0 0 1rem 0;}
.ym-card {background:#fff; border:1px solid #dfe6e3; border-radius:16px; padding:14px 16px 12px; margin:0 0 10px 0;
          box-shadow: 0 1px 2px rgba(18,59,50,.04);}
.ym-ko {font-size:.85rem; color:#5c6b66; margin-bottom:2px;}
.ym-native {font-size:1.75rem; line-height:1.25; font-weight:650; color:#123b32; word-break:break-word; margin:2px 0;}
.ym-native[dir=rtl] {text-align:right;}
.ym-roman {font-size:.95rem; color:#2f3d39; font-family: ui-monospace, Menlo, Consolas, monospace;}
.ym-hangul {font-size:1.1rem; font-weight:700; color:#1f6e4e; margin-top:2px;}
.ym-tip {font-size:.82rem; color:#5c6b66; margin-top:6px; padding-top:6px; border-top:1px dashed #e3e8e5;}
.ym-group {display:inline-block; font-size:.78rem; font-weight:700; letter-spacing:.04em; color:#1f6e4e;
           background:#e6f1eb; border-radius:999px; padding:3px 10px; margin:10px 0 6px 0;}
.ym-guide {background:#fff; border:1px solid #dfe6e3; border-radius:16px; padding:6px 18px 10px;}
.ym-guide h4 {font-size:.82rem; color:#5c6b66; font-weight:700; margin:12px 0 2px; letter-spacing:.02em;}
.ym-guide p {font-size:1.05rem; line-height:1.55; margin:0 0 2px 0;}
.ym-guide .ko {font-size:.85rem; color:#7a8683; margin:0 0 4px 0;}
.ym-basis {font-size:.78rem; color:#7a8683; margin-top:10px;}
.ym-tag {display:inline-block; font-size:.75rem; padding:2px 8px; border-radius:6px; margin-right:6px; font-weight:600;}
.ym-tag.drug {background:#e8eef9; color:#1f3f8a;} .ym-tag.htfs {background:#eef7e6; color:#2f6b1f;} .ym-tag.cosm {background:#fbeef2; color:#8a1f4a;}
div[data-testid="stAudio"] {margin-top:6px;}
.ym-pinyin-row {display:flex; flex-wrap:wrap; gap:2px 4px; background:#f1f4f2; border-radius:8px; padding:10px 12px;}
.ym-pin-cell {display:flex; flex-direction:column; align-items:center; min-width:1.4em;}
.ym-pin-py {font-size:.72rem; color:#1f6e4e; font-family: ui-monospace, Menlo, Consolas, monospace; white-space:nowrap;}
.ym-pin-ch {font-size:1.15rem; color:#1c2a24; line-height:1.1;}
ruby rt {font-size:.6em; color:#5c6b66;}
</style>
<div class="ym-brand"><h1>💊 약말</h1><span>외국인 고객 응대 · 복약안내 + 약사 발음 도우미</span></div>
<p class="ym-sub">식약처 공공데이터를 근거로 고객 언어 안내문을 만들고, 약사가 직접 말할 핵심 표현을 발음과 함께 제시합니다.</p>
""", unsafe_allow_html=True)

if not DATA_GO_KR_KEY or not GEMINI_API_KEY:
    st.warning("Secrets에 DATA_GO_KR_KEY, GEMINI_API_KEY를 넣어주세요.")
    st.stop()

if "my_products" not in st.session_state:
    with st.spinner("우리 약국 목록 확인 중… (처음 한 번만)"):
        st.session_state["my_products"] = resolve_my_items(load_my_from_file())
    st.session_state["my_dirty"] = False
render_my_sidebar()
my_keys = {my_key(x) for x in st.session_state["my_products"]}

col_m, col_l = st.columns([3, 1])
mode = col_m.radio("모드", ["제품 검색", "🎤 고객 말 듣기"], horizontal=True, label_visibility="collapsed")
language = col_l.selectbox("고객 언어", list(LANGUAGES))
flags = tuple(st.multiselect("고객 특이사항 (선택)", FLAGS, placeholder="할랄, 비건 등 해당되면 선택"))

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
    qkey = query.replace(" ", "")
    for x in st.session_state["my_products"]:
        if x.get("kind") == "화장품(일반)" and (qkey in x["name"].replace(" ", "") or x["name"].replace(" ", "") in qkey):
            if not any(r["name"] == x["name"] for r in results):
                results.append({"kind": "화장품(일반)", "name": x["name"], "maker": x.get("maker", ""), "raw": {}})
    for kind, msg in errors.items():
        st.caption(f"⚠️ {kind} 조회 실패: {msg[:120]}")

    if not results:
        st.info("의약품·건강기능식품·기능성화장품에서 찾지 못했어요.")
        maker_g = st.text_input("제조사 (선택)", key="gc_maker")
        if st.button(f"🧴 '{query}'를 일반 화장품으로 조회", key="gc_go"):
            st.session_state["gc_pick"] = {"kind": "화장품(일반)", "name": query.strip(), "maker": maker_g.strip(), "raw": {}}
        if st.session_state.get("gc_pick", {}).get("name") == query.strip():
            results = [st.session_state["gc_pick"]]
    else:
        results.sort(key=lambda r: my_key(r) not in my_keys)
        labels = [f"{'⭐ ' if my_key(r) in my_keys else ''}[{r['kind']}] {r['name']} · {r['maker']}" for r in results]
        pick = st.radio("제품 선택", labels)
        chosen = results[labels.index(pick)]
        raw = chosen["raw"]
        if my_key(chosen) in my_keys:
            if st.button("✕ 우리 약국에서 제거", key="rm_chosen"):
                my_remove(my_key(chosen))
                st.rerun()
        else:
            if st.button("⭐ 우리 약국에 추가", key="add_chosen"):
                my_add(chosen)
                st.rerun()

        if chosen["kind"] == "의약품":
            easy = easy_by_seq(raw.get("ITEM_SEQ", ""))
            source_text = build_drug_text(raw, easy)
            basis = "식약처 의약품 허가정보" + (" + e약은요" if easy else "")
            if "전문" in str(raw.get("ETC_OTC_CODE", "")):
                st.error("전문의약품입니다. 처방전 없이 판매할 수 없어요.")
        elif chosen["kind"] == "화장품(일반)":
            easy = None
            st.warning("⚠️ 식약처 기능성 인증이 없는 일반 화장품입니다. 효능은 회사 자체 시험/표시 정보이며 식약처 인증이 아닙니다.")
            ingr_text = ""
            if naver_ok() and st.button("🔎 네이버에서 전성분·시험결과 찾기", key=f"nv_gc_{chosen['name']}"):
                st.session_state["nvgc_for"] = chosen["name"]
            claims_text = ""
            if naver_ok() and st.session_state.get("nvgc_for") == chosen["name"]:
                with st.spinner("네이버 검색 중…"):
                    try:
                        w = naver_claims(chosen["name"], chosen["maker"])
                    except Exception as e:
                        w = {"found": False, "note": str(e)[:200]}
                if w.get("found"):
                    ingr_text = w.get("ingredients", "") or ""
                    claims_text = w.get("claims", "") or ""
                    st.info(f"웹 출처({w.get('source','')}) — 포장·공식 자료와 대조 필요\n\n"
                            + (f"시험 결과: {claims_text}\n\n" if claims_text else "")
                            + (f"전성분: {ingr_text[:300]}{'…' if len(ingr_text)>300 else ''}" if ingr_text else ""))
                else:
                    st.caption(f"웹에서 찾지 못했어요. {w.get('note','')}")
            up_i = st.text_area("또는 전성분 붙여넣기", height=60, placeholder="포장의 전성분표")
            up_c = st.text_area("또는 시험 결과·특징 붙여넣기", height=60, placeholder="예: 12주 사용 후 피부 요철 개선율 87% (자사 시험)")
            ingr_text = up_i.strip() or ingr_text
            claims_text = up_c.strip() or claims_text
            source_text = build_general_cosm_text(chosen["name"], chosen["maker"], ingr_text, claims_text)
            basis = "제조사 표시·웹 정보 (식약처 인증 아님)"
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
                out = generate(source_text, language, flags)
            except Exception as e:
                st.error(f"생성 실패: {e}")
                st.stop()

        tts_code = LANGUAGES[language][0]
        left, right = st.columns(2)

        with left:
            st.subheader(f"🧾 고객용 안내 · {language}")
            if chosen["kind"] == "의약품" and easy and easy.get("itemImage"):
                st.image(easy["itemImage"], width=200, caption="낱알 모양")
            g = out.get("patient_guide", {}) or {}
            head = "이 약은" if chosen["kind"] == "의약품" else "이 제품은"
            how = {"의약품": "복용법", "건강기능식품": "섭취방법"}.get(chosen["kind"], "사용법")
            rtl = ' dir="rtl"' if language == "아랍어" else ""
            is_zh = language in ("중국어(보통화)", "중국어(번체·대만)", "광둥어")
            sections = []
            for label, k in [(head, "what_it_is"), (how, "how_to_take"), ("주의사항", "cautions"),
                             ("약사 상담이 필요한 경우", "see_pharmacist_if")]:
                v = g.get(k)
                tr, ko, pinyin_pairs = pair(v)
                if not tr:
                    continue
                tr_html = tr if language == "일본어" else esc(tr)
                sections.append((label, tr_html, ko, pinyin_pairs))

            html = ['<div class="ym-guide">']
            st.markdown("".join(html), unsafe_allow_html=True)
            for label, tr_html, ko, pinyin_pairs in sections:
                part = [f"<h4>{esc(label)}</h4>", f"<p{rtl}>{tr_html}</p>"]
                if ko:
                    part.append(f'<p class="ko">{esc(ko)}</p>')
                st.markdown("".join(part), unsafe_allow_html=True)
                if is_zh and pinyin_pairs:
                    with st.expander("拼音 병음 보기"):
                        cells = "".join(
                            f'<div class="ym-pin-cell"><div class="ym-pin-py">{esc(x.get("p",""))}</div>'
                            f'<div class="ym-pin-ch">{esc(x.get("c",""))}</div></div>'
                            for x in pinyin_pairs if isinstance(x, dict) and x.get("c")
                        )
                        st.markdown(f'<div class="ym-pinyin-row">{cells}</div>', unsafe_allow_html=True)
            html = []
            html.append(f'<div class="ym-basis">근거: {esc(basis)} · AI 생성 안내, 약사 확인 후 제공</div></div>')
            st.markdown("".join(html), unsafe_allow_html=True)

        with right:
            st.subheader("🗣️ 약사가 직접 말하기")
            render_phrases(out.get("key_phrases"), language)

        if chosen["kind"] == "의약품":
            render_dur(raw, language)
        render_country(out.get("country_notes") or {}, language)
        render_reviews(chosen["name"], chosen["kind"])
