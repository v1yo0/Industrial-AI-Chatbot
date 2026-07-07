import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["API_TRANSPORT"] = "rest"
import sys
import re
import json
import unicodedata
import time
import html
import asyncio
import uvicorn

# Đảm bảo mã hóa UTF-8 cho console trên Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import google
import google.generativeai as genai
import google.api_core.exceptions
import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchAny

# Nạp cấu hình từ file .env
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BACKEND_DIR, ".env")
load_dotenv(ENV_PATH, override=True)

# ============================================================
# CÁC HÀM TIỆN ÍCH
# ============================================================

def remove_vietnamese_accents(input_str: str) -> str:
    """Loại bỏ dấu tiếng Việt để so khớp dễ dàng hơn"""
    if not input_str:
        return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return only_ascii.replace('đ', 'd').replace('Đ', 'D').lower()


def normalize_sku(text):
    """Chuẩn hóa mã SKU: loại bỏ ký tự đặc biệt, chỉ giữ chữ và số"""
    return re.sub(r'[^A-Za-z0-9]', '', text).lower()


def extract_sku_tokens(query):
    """Trích xuất các token có dạng mã SKU từ câu hỏi người dùng"""
    tokens = re.findall(r'[A-Za-z0-9\-_]+', query)
    sku_tokens = []
    for t in tokens:
        has_digit = any(c.isdigit() for c in t)
        has_alpha = any(c.isalpha() for c in t)
        if len(t) >= 3 and (t.count('-') > 0 or (has_digit and has_alpha)):
            sku_tokens.append(t.lower())
    return sku_tokens


def is_broad_model_token(token: str) -> bool:
    """
    Nhận diện token dạng dòng/series như E5CC, không phải model cụ thể.
    Series: E5CC, TM, ACT20C (ngắn, không có dấu '-' hoặc chỉ 1 phần)
    Model: E5CC-QX3ASM-001 (dài, có nhiều phần phân cách bởi '-')
    """
    # Nếu token chứa dấu '-' → có thể là model cụ thể
    if '-' in token:
        parts = token.split('-')
        # Nếu chỉ 2 phần và phần sau ngắn → vẫn có thể là series (VD: TM-T4Y)
        if len(parts) >= 3:
            return False  # Model cụ thể: E5CC-QX3ASM-001
        # 2 phần: kiểm tra tổng chiều dài
        if sum(len(p) for p in parts) > 8:
            return False  # Model cụ thể đủ dài
    
    normalized = normalize_sku(token)
    return len(normalized) <= 7  # Series thường <= 7 ký tự alphanumeric


def is_generic_type(t: str) -> bool:
    """Kiểm tra xem từ khóa loại thiết bị có quá chung chung hay không"""
    if not t:
        return True
    t_norm = remove_vietnamese_accents(t).strip().lower()
    generic_words = {
        "san pham", "thiet bi", "mat hang", "hang", "loai", "dong",
        "device", "product", "item", "sanpham", "thietbi"
    }
    return t_norm in generic_words


def tokenize_vietnamese(text: str) -> list[str]:
    """Tokenize tiếng Việt cho BM25: loại bỏ stopwords, tạo unigrams và bigrams."""
    if not text:
        return []
    text_norm = remove_vietnamese_accents(text).lower()
    tokens = re.findall(r"[a-z0-9]+", text_norm)
    stopwords = {
        "toi", "cho", "cua", "la", "co", "nhung", "nao", "giup",
        "va", "de", "trong", "voi", "duoc", "se", "da", "dang",
        "mot", "cac", "nay", "do", "tu", "len", "ra", "vao",
        "the", "ma", "thi", "neu", "con", "nhu", "boi", "tai",
        "hay", "hoac", "noi", "ve", "di", "an", "chi", "o",
    }
    unigrams = [t for t in tokens if len(t) >= 2 and t not in stopwords]
    bigrams = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams)-1)]
    return unigrams + bigrams


def xml_text(value) -> str:
    return html.escape(str(value or ""), quote=False)


def fix_markdown_table(text: str) -> str:
    """Tự động sửa lỗi dòng phân cách (delimiter row) và lỗi dư cột ở dòng tiêu đề."""
    if not text or "|" not in text:
        return text
    lines = text.split('\n')
    for i in range(1, len(lines)):
        # Nếu dòng hiện tại là dòng phân cách (chỉ chứa |, -, :, space) và dòng trước đó có |
        if re.match(r'^[\s\|]+[\-\:\s\|]+[\s\|]+$', lines[i]) and '|' in lines[i-1]:
            header_line = lines[i-1].strip()
            header_parts = [c for c in header_line.split('|') if c.strip() != '']
            header_cols = len(header_parts)
            
            # Kiểm tra dòng dữ liệu đầu tiên (nếu có) để biết số cột thực tế
            data_cols = header_cols
            if i + 1 < len(lines) and '|' in lines[i+1]:
                data_line = lines[i+1].strip()
                data_cols = len([c for c in data_line.split('|') if c.strip() != ''])
            
            # Nếu dòng tiêu đề bị AI tự ý tách Hãng thành cột riêng (5 cột) so với dữ liệu (3 cột)
            if header_cols == 5 and data_cols == 3:
                # Gộp cột 2+3 (SP1 + Hãng1) và 4+5 (SP2 + Hãng2)
                new_h2 = f"{header_parts[1].strip()} ({header_parts[2].strip()})"
                new_h3 = f"{header_parts[3].strip()} ({header_parts[4].strip()})"
                lines[i-1] = f"| {header_parts[0].strip()} | {new_h2} | {new_h3} |"
                header_cols = 3
            
            # Cấu trúc lại dòng phân cách với số lượng cột bằng chính xác dòng tiêu đề
            if header_cols > 0:
                lines[i] = '|' + '---|' * header_cols
    return '\n'.join(lines)


def split_description(description: str) -> list[str]:
    description_str = str(description or "").strip()
    if not description_str:
        return []
    if "|" in description_str:
        return [part.strip() for part in description_str.split("|") if part.strip()]
    if "\n" in description_str:
        return [part.strip() for part in description_str.split("\n") if part.strip()]
    return [description_str]


def compute_keyword_score(text: str, query: str) -> float:
    """Tính điểm tần suất từ khóa đơn giản phục vụ tiền xử lý."""
    if not query or not text:
        return 0.0
    text_norm = remove_vietnamese_accents(text).lower()
    query_norm = remove_vietnamese_accents(query).lower()
    words = [
        w for w in re.findall(r"[a-z0-9]+", query_norm)
        if len(w) >= 2 and w not in {
            "toi", "cho", "cua", "hang", "san", "pham", "thiet", "bi",
            "gia", "bao", "nhieu", "tim", "xem", "tu", "van", "la",
            "co", "nhung", "nao", "giup"
        }
    ]
    score = 0.0
    for w in words:
        if re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", text_norm):
            score += 1.0
        elif w in text_norm:
            score += 0.35
    return score


def build_relevant_description(description: str, query: str, max_chars: int = 1600) -> str:
    """Chọn các dòng mô tả liên quan nhất với câu hỏi."""
    lines = split_description(description)
    if not lines:
        return ""

    original = " | ".join(lines)
    if len(original) <= max_chars:
        return original

    scored = [(compute_keyword_score(line, query), idx, line) for idx, line in enumerate(lines)]
    selected_indices = {0}
    for score, idx, _ in sorted(scored, key=lambda item: item[0], reverse=True)[:8]:
        if score > 0:
            selected_indices.add(idx)

    selected_lines = [lines[idx] for idx in sorted(selected_indices)]
    snippet = " | ".join(selected_lines)
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "..."
    return snippet


def extract_technical_specs(description: str) -> str:
    """Trích xuất thông số kỹ thuật từ mô tả sản phẩm."""
    if not description:
        return "Không có mô tả chi tiết."
    
    description_str = str(description).strip()
    lines = []
    if "|" in description_str:
        lines = [part.strip() for part in description_str.split("|") if part.strip()]
    elif "\n" in description_str:
        lines = [part.strip() for part in description_str.split("\n") if part.strip()]
    elif "◎" in description_str:
        lines = [part.strip() for part in description_str.split("◎") if part.strip()]
    else:
        return description_str[:1200]
            
    specs = []
    for line in lines:
        if ":" in line:
            key, val = line.split(":", 1)
            specs.append(f"- **{key.strip()}**: {val.strip()}")
        elif "：" in line:
            key, val = line.split("：", 1)
            specs.append(f"- **{key.strip()}**: {val.strip()}")
        else:
            specs.append(f"- {line}")
            
    if not specs:
        return description_str[:1200]
        
    return "\n".join(specs)


def build_evidence_xml(prod: dict, query: str, mode: str = "detail") -> str:
    """
    Tạo context XML cho sản phẩm với evidence-based packing.
    - Structured specs đặt lên đầu (dễ đọc, giảm hallucination)
    - Mô tả liên quan đến query đặt giữa
    - Mô tả phụ đặt cuối hoặc cắt bỏ
    
    mode: 'detail' (đầy đủ), 'list' (gọn), 'compare' (thông số)
    """
    name = xml_text(prod['device'])
    brand = xml_text(prod.get('brand', prod['company']))
    distributor = xml_text(prod['company'])
    category = xml_text(prod.get('category', ''))
    series = xml_text(prod.get('series', ''))
    price = xml_text(prod['price_formatted'])
    
    specs = prod.get('specs', {})
    
    # Phần evidence: specs liên quan đến query đặt lên đầu
    evidence_lines = []
    other_spec_lines = []
    
    if specs:
        query_norm = remove_vietnamese_accents(query).lower()
        for key, val in specs.items():
            key_norm = remove_vietnamese_accents(key).lower()
            val_norm = remove_vietnamese_accents(val).lower()
            line = f"    {xml_text(key)}: {xml_text(val)}"
            
            query_words = [w for w in re.findall(r'[a-z0-9]+', query_norm) if len(w) >= 2]
            is_relevant = any(w in key_norm or w in val_norm for w in query_words)
            
            if is_relevant:
                evidence_lines.append(line)
            else:
                other_spec_lines.append(line)
    
    xml_parts = [
        f"  <name>{name}</name>",
        f"  <brand>{brand}</brand>",
        f"  <distributor>{distributor}</distributor>",
        f"  <category>{category}</category>",
    ]
    
    if series:
        xml_parts.append(f"  <series>{series}</series>")
    
    xml_parts.append(f"  <price>{price}</price>")
    
    # Evidence specs (liên quan đến query) lên đầu
    if evidence_lines:
        xml_parts.append(f"  <evidence_specs>")
        xml_parts.extend(evidence_lines)
        xml_parts.append(f"  </evidence_specs>")
    
    # Other specs
    if mode in ("detail", "compare"):
        max_other = 15 if mode == "detail" else 10
        if other_spec_lines:
            xml_parts.append(f"  <other_specs>")
            xml_parts.extend(other_spec_lines[:max_other])
            xml_parts.append(f"  </other_specs>")
    
    # Description (chỉ cho detail/list/compare, và chỉ nếu specs ít)
    if mode == "detail" and len(specs) < 3:
        desc = build_relevant_description(prod.get('description', ''), query, max_chars=1600)
        if desc:
            xml_parts.append(f"  <description>{xml_text(desc)}</description>")
    elif mode == "list":
        desc = build_relevant_description(prod.get('description', ''), query, max_chars=500)
        if desc and len(specs) < 3:
            xml_parts.append(f"  <description>{xml_text(desc)}</description>")
    elif mode == "compare":
        desc = build_relevant_description(prod.get('description', ''), query, max_chars=800)
        if desc and len(specs) < 3:
            xml_parts.append(f"  <description>{xml_text(desc)}</description>")
    
    return "\n".join(xml_parts)


# ============================================================
# FASTAPI APP INIT
# ============================================================

app = FastAPI(title="Industrial Devices RAG Chatbot Backend")

allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5500,http://127.0.0.1:5500",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GLOBAL STATE
# ============================================================

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_JSON_PATH = os.path.join(BACKEND_DIR, "vector_store", "metadata.json")
CHUNK_MAP_PATH = os.path.join(BACKEND_DIR, "vector_store", "chunk_map.json")

qdrant_client_instance = None
metadata = None
SYNONYM_MAP = {
    ("động cơ", "motor", "dong co"): ["động cơ", "motor", "servo motor", "geared motor", "stepping motor"],
    ("can nhiệt", "cặp nhiệt", "can nhiet", "cap nhiet", "thermocouple"): ["can nhiệt", "cặp nhiệt", "thermo", "e52"],
    ("bộ điều khiển nhiệt độ", "bo dieu khien nhiet do", "đồng hồ nhiệt độ", "dong ho nhiet do", "temperature controller"): 
        ["bộ điều khiển nhiệt độ", "bộ điều khiển trương trình bước nhiệt", "đồng hồ nhiệt độ", "bộ ghi dữ liệu nhiệt độ"],
    ("bộ điều khiển", "bo dieu khien", "controller"): ["bộ điều khiển", "controller", "driver điều khiển"],
    ("cảm biến nhiệt", "cam bien nhiet"): ["cảm biến nhiệt", "can nhiệt", "cặp nhiệt", "đầu dò nhiệt"],
    ("cảm biến tiệm cận", "cam bien tiem can", "tiệm cận", "tiem can", "proximity"): ["cảm biến tiệm cận", "cảm biến cảm ứng"],
    ("cảm biến quang", "cam bien quang", "photoelectric"): ["cảm biến quang", "cảm biến quang điện"],
    ("cảm biến áp suất", "cam bien ap suat", "pressure"): ["cảm biến áp suất"],
    ("cảm biến", "cam bien", "sensor"): ["cảm biến", "sensor"],
    ("máy sấy", "may say", "dryer"): ["máy sấy"],
    ("máy tuốt", "may tuot"): ["máy tuốt", "mira", "cosmic"],
    ("máy hút bụi", "may hut bui", "vacuum"): ["máy hút bụi"],
    ("kích", "thủy lực", "kich thuy luc", "hydraulic", "jack", "con rùa", "con thua"): ["kích", "kích thủy lực", "thủy lực", "con rùa", "hydraulic", "hydraulic jack"],
    ("máy ghi", "may ghi", "recorder"): ["máy ghi", "bộ ghi"],
    ("van", "valve", "van điện từ"): ["van", "van điện từ"],
    ("xi lanh", "cylinder"): ["xi lanh", "xi lanh khí"],
    ("khúc xạ kế", "khuc xa ke", "refractometer"): ["khúc xạ kế"],
    ("kìm",): ["kìm"],
    ("găng tay", "gang tay"): ["găng tay"],
    ("quạt", "quat", "fan"): ["quạt"],
    ("hộp số", "hop so", "gearbox"): ["hộp số", "gearbox", "geared"],
    ("bơm", "bom", "pump"): ["bơm", "ống bơm"],
    ("đầu nối", "dau noi", "connector", "terminal"): ["đầu nối"],
    ("biến tần", "bien tan", "inverter"): ["biến tần", "inverter"],
    ("plc",): ["plc", "bộ lập trình"],
    ("relay", "rơ le", "ro le"): ["relay", "rơ le"],
    ("nguồn", "nguon", "power supply"): ["nguồn", "nguồn cấp", "power supply"],
    ("bộ chuyển đổi tín hiệu", "signal converter", "signal isolator"): ["bộ chuyển đổi tín hiệu", "signal converter"],
    ("quần áo", "áo", "quần", "trang phục", "bảo hộ", "giày"): ["quần áo", "áo", "quần", "trang phục", "bảo hộ", "giày", "phòng sạch"],
}
ALL_PRODUCT_TYPES = []
chunk_map = None
product_to_chunks = None
embedding_model = None
reranker_model = None
bm25_index = None
bm25_corpus_tokens = None
gemini_model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Caching & Inverted Indexes
query_embedding_cache = {}
search_result_cache = {}
brand_index = {}
category_index = {}
series_index = {}
sku_index = {}

# ============================================================
# GEMINI API KEY MANAGER
# ============================================================

class GeminiKeyManager:
    def __init__(self):
        raw_keys = os.environ.get("GEMINI_API_KEYS", "")
        self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        
        # Tương thích ngược với GEMINI_API_KEY đơn lẻ
        if not self.api_keys or self.api_keys == ["YOUR_GEMINI_API_KEY"]:
            single_key = os.environ.get("GEMINI_API_KEY")
            if single_key and single_key != "YOUR_GEMINI_API_KEY":
                self.api_keys = [single_key]
                
        # Loại bỏ các key mẫu nếu có
        self.api_keys = [k for k in self.api_keys if k != "YOUR_GEMINI_API_KEY"]
                
        self.current_index = 0
        self.key_cooldowns = {key: 0.0 for key in self.api_keys}
        
    def get_working_key(self) -> str:
        if not self.api_keys:
            return None
            
        now = time.time()
        num_keys = len(self.api_keys)
        
        for _ in range(num_keys):
            key = self.api_keys[self.current_index]
            if now >= self.key_cooldowns.get(key, 0.0):
                return key
            self.current_index = (self.current_index + 1) % num_keys
            
        min_cooldown_key = min(self.api_keys, key=lambda k: self.key_cooldowns.get(k, 0.0))
        wait_time = self.key_cooldowns[min_cooldown_key] - now
        if wait_time > 0:
            print(f"[KeyManager] Tất cả các API Key đều bị giới hạn 429. Đang chờ {wait_time:.1f} giây...")
            time.sleep(wait_time)
        return min_cooldown_key
        
    def mark_key_limit(self, key: str, cooldown_seconds: float = 30.0):
        if key in self.key_cooldowns:
            masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "Key"
            print(f"[KeyManager] API Key {masked} bị lỗi 429. Phạt cooldown {cooldown_seconds}s.")
            self.key_cooldowns[key] = time.time() + cooldown_seconds
            self.current_index = (self.api_keys.index(key) + 1) % len(self.api_keys)
            
    def configure_sdk(self) -> str:
        key = self.get_working_key()
        if key:
            genai.configure(api_key=key)
        return key

key_manager = GeminiKeyManager()

if not key_manager.api_keys:
    print("=========================================================================")
    print("CẢNH BÁO: Chưa cấu hình GEMINI_API_KEY hoặc GEMINI_API_KEYS trong file .env!")
    print("Vui lòng cập nhật file .env để chatbot có thể hoạt động.")
    print("=========================================================================")
else:
    print(f"Hệ thống xoay vòng API đã sẵn sàng với {len(key_manager.api_keys)} API Key:")
    for idx, k in enumerate(key_manager.api_keys):
        masked_key = k[:6] + "..." + k[-4:] if len(k) > 10 else "ShortKey"
        print(f"  - Key #{idx + 1}: {masked_key}")
    key_manager.configure_sdk()


# ============================================================
# KHỞI ĐỘNG BACKEND
# ============================================================

def init_backend():
    global qdrant_client_instance, metadata, chunk_map, product_to_chunks, embedding_model, reranker_model, bm25_index, bm25_corpus_tokens
    global brand_index, category_index, series_index, sku_index
    
    print("--- ĐANG KHỞI ĐỘNG BACKEND ---")
    
    if not os.path.exists(METADATA_JSON_PATH):
        print("Cảnh báo: Chưa tìm thấy Metadata. Vui lòng chạy ingest.py trước.")
        return
        
    print("Đang khởi tạo kết nối Qdrant và tải Metadata...")
    try:
        qdrant_client_instance = qdrant_client.QdrantClient(url="http://localhost:6333")
        # Kiểm tra kết nối
        qdrant_client_instance.get_collections()
    except Exception as e:
        print(f"Cảnh báo: Không thể kết nối Qdrant Docker. Lỗi: {e}")
        
    with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Đã tải thành công {len(metadata)} sản phẩm.")
    
    # Tự động tạo danh sách tất cả các loại thiết bị (từ DB + từ điển đồng nghĩa)
    global ALL_PRODUCT_TYPES
    db_cats = set(prod.get("category", "") for prod in metadata if prod.get("category"))
    synonym_keys = set()
    for keys in SYNONYM_MAP.keys():
        synonym_keys.update(keys)
    ALL_PRODUCT_TYPES = sorted(list(db_cats.union(synonym_keys)), key=len, reverse=True)
    
    print("Đang xây dựng Inverted Indexes...")
    for idx, prod in enumerate(metadata):
        b_norm = remove_vietnamese_accents(prod.get("brand", "")).lower()
        if b_norm:
            brand_index.setdefault(b_norm, set()).add(idx)
        c_norm = remove_vietnamese_accents(prod.get("company", "")).lower()
        if c_norm:
            brand_index.setdefault(c_norm, set()).add(idx)
        cat_norm = remove_vietnamese_accents(prod.get("category", "")).lower()
        if cat_norm:
            category_index.setdefault(cat_norm, set()).add(idx)
        ser_norm = normalize_sku(prod.get("series", ""))
        if ser_norm:
            series_index.setdefault(ser_norm, set()).add(idx)
        skus = [normalize_sku(t) for t in extract_sku_tokens(prod.get("device", ""))]
        for s in skus:
            if s:
                sku_index.setdefault(s, set()).add(idx)
    print("Inverted Indexes đã sẵn sàng.")
    
    # Tải chunk map (mapping chunk_idx → product_idx)
    if os.path.exists(CHUNK_MAP_PATH):
        with open(CHUNK_MAP_PATH, "r", encoding="utf-8") as f:
            chunk_map = json.load(f)
        # Build reverse map: product_idx → [chunk_indices]
        product_to_chunks = {}
        for cm in chunk_map:
            prod_idx = cm["product_idx"]
            chunk_idx = cm["chunk_idx"]
            if prod_idx not in product_to_chunks:
                product_to_chunks[prod_idx] = []
            product_to_chunks[prod_idx].append(chunk_idx)
        print(f"Đã tải chunk map: {len(chunk_map)} chunks cho {len(product_to_chunks)} sản phẩm.")
    else:
        print("Cảnh báo: Không tìm thấy chunk_map.json. Sử dụng chế độ tương thích cũ (1 embedding = 1 sản phẩm).")
        chunk_map = None
        product_to_chunks = None
    
    print("Đang tải mô hình Embedding (intfloat/multilingual-e5-small)...")
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")
    
    # Xây dựng BM25 Index — bao gồm series + specs cho matching kỹ thuật
    print("Đang xây dựng BM25 Index (có series + specs)...")
    bm25_corpus_tokens = []
    for prod in metadata:
        series = prod.get('series', '')
        specs = prod.get('specs', {})
        # Lấy specs values ngắn gọn cho BM25
        specs_vals = ' '.join(list(specs.values())[:8]) if specs else ''
        text = (
            f"{prod.get('brand', '')} {prod.get('category', '')} "
            f"{series} {prod.get('device', '')} "
            f"{specs_vals} "
            f"{prod.get('description', '')[:400]}"
        )
        bm25_corpus_tokens.append(tokenize_vietnamese(text))
    bm25_index = BM25Okapi(bm25_corpus_tokens)
    print("BM25 Index đã sẵn sàng.")
    
    # Tải Cross-Encoder Reranker
    print("Đang tải Cross-Encoder Reranker (có thể mất 30-60s lần đầu)...")
    try:
        reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
        print("Cross-Encoder Reranker đã sẵn sàng.")
    except Exception as e:
        print(f"Cảnh báo: Không tải được Reranker: {e}. Hệ thống vẫn hoạt động không có reranking.")
        reranker_model = None
    
    print(f"Sử dụng mô hình LLM: {gemini_model_name}")
    print("--- BACKEND ĐÃ SẴN SÀNG! ---")


init_backend()


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/api/status")
async def status_endpoint():
    return {
        "status": "online", 
        "model": gemini_model_name,
        "gemini_api_configured": bool(key_manager.api_keys),
        "reranker_loaded": reranker_model is not None
    }


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int = 6


# ============================================================
# HÃNG SẢN XUẤT VÀ ĐỒNG NGHĨA LOẠI THIẾT BỊ
# ============================================================

KNOWN_BRANDS = [
    "toho electronics", "oriental motor", "sanyo denki", "weidmuller",
    "yodogawa", "komax", "omron", "jumo", "huba control",
    "master flex", "masterflex", "balluff", "endress hauser",
    "keyence", "mitsubishi", "siemens", "schneider", "abb",
    "panasonic", "fuji electric", "autonics", "idec", "delta",
    "sick", "ifm", "turck", "pepperl fuchs", "banner",
    "azbil", "yokogawa", "honeywell", "phoenix contact",
    "mira", "cosmic", "daikin", "smc", "ckd", "festo",
    "nb", "as one", "renishaw", "nachi", "nsk", "thk",
    "hiwin", "iai", "yaskawa", "fanuc", "beckhoff",
]


def get_synonyms_for_type(type_query: str) -> list[str]:
    """Trả về danh sách từ đồng nghĩa cho một loại thiết bị"""
    if not type_query:
        return []
    t = type_query.strip().lower()
    
    t_norm = remove_vietnamese_accents(t)
    for keys, synonyms in SYNONYM_MAP.items():
        for key in keys:
            key_norm = remove_vietnamese_accents(key)
            if key in t or key_norm in t_norm:
                return synonyms
    
    return [t]


# ============================================================
# HEURISTIC METADATA EXTRACTION
# ============================================================

def extract_price_filter(query: str) -> dict:
    """Trích xuất khoảng giá từ câu hỏi bằng Regex."""
    q = remove_vietnamese_accents(query).lower()
    
    multiplier = {"trieu": 1000000, "tr": 1000000, "k": 1000, "nghin": 1000, "ngan": 1000, "vnd": 1, "dong": 1}
    
    # 1. Từ X đến Y
    match_range = re.search(r'(?:tu\s+)?(\d+(?:\.\d+)?)\s*(?:-|den)\s*(\d+(?:\.\d+)?)\s*(trieu|tr|k|nghin|ngan|vnd|dong)', q)
    if match_range:
        v1, v2, unit = float(match_range.group(1)), float(match_range.group(2)), match_range.group(3)
        mult = multiplier.get(unit, 1)
        return {"min": int(v1 * mult), "max": int(v2 * mult)}
        
    # 2. Dưới X
    match_under = re.search(r'(duoi|nho hon|thap hon|<|re hon)\s*(\d+(?:\.\d+)?)\s*(trieu|tr|k|nghin|ngan|vnd|dong)', q)
    if match_under:
        v, unit = float(match_under.group(2)), match_under.group(3)
        mult = multiplier.get(unit, 1)
        return {"min": None, "max": int(v * mult)}
        
    # 3. Trên X
    match_over = re.search(r'(tren|lon hon|cao hon|>|dat hon)\s*(\d+(?:\.\d+)?)\s*(trieu|tr|k|nghin|ngan|vnd|dong)', q)
    if match_over:
        v, unit = float(match_over.group(2)), match_over.group(3)
        mult = multiplier.get(unit, 1)
        return {"min": int(v * mult), "max": None}
        
    return None

def extract_metadata_from_text(text: str) -> dict:
    """Trích xuất company, product_type, model_code từ text bằng heuristics"""
    text_lower = text.lower()
    text_norm = remove_vietnamese_accents(text_lower)
    
    model_code = []
    skus = extract_sku_tokens(text)
    if skus:
        model_code = [normalize_sku(s) for s in skus]
        
    company = None
    sorted_brands = sorted(KNOWN_BRANDS, key=len, reverse=True)
    for brand in sorted_brands:
        brand_norm = remove_vietnamese_accents(brand)
        if brand_norm in text_norm:
            company = brand
            break
            
    product_type = None
    for t_kw in ALL_PRODUCT_TYPES:
        t_kw_norm = remove_vietnamese_accents(t_kw)
        if f" {t_kw_norm} " in f" {text_norm} " or f" {t_kw} " in f" {text_lower} ":
            product_type = t_kw
            break
            
    price_filter = extract_price_filter(text)
            
    return {
        "company": company,
        "product_type": product_type,
        "model_code": model_code,
        "price_filter": price_filter
    }

def determine_action_heuristically(query: str, filters: dict) -> tuple[str, bool]:
    """Xác định action và clarification_needed bằng Python heuristics."""
    q_norm = remove_vietnamese_accents(query).lower()
    
    chitchat_patterns = [
        "chao", "hello", "hi ", "xin chao", "chao ban", "ban la ai", 
        "tro ly gi", "giup gi", "cam on", "thank", "ok"
    ]
    if len(q_norm.split()) <= 4 and any(p in q_norm for p in chitchat_patterns):
        return "chitchat", False
        
    count_patterns = ["bao nhieu san pham", "so luong", "thong ke", "dem", "tong cong", "co tat ca", "co bao nhieu"]
    price_patterns = ["gia", "tien", "bao nhieu tien", "gia bao nhieu", "bao nhieu k", "gia ca", "gia the nao", "may tien"]
    
    is_count_query = any(p in q_norm for p in count_patterns)
    is_price_inquiry = any(p in q_norm for p in ["gia bao nhieu", "bao nhieu tien", "gia the nao", "may tien"])
    
    # Nếu câu hỏi đếm số lượng, ưu tiên action count (ngay cả khi có đề cập giá)
    if is_count_query and not is_price_inquiry:
        return "count", False
        
    compare_patterns = ["so sanh", "khac nhau", "khac gi", "tot hon", "nen chon cai nao", "uu diem", "nhuoc diem"]
    if any(p in q_norm for p in compare_patterns):
        return "compare", False
        
    list_patterns = [
        "co nhung", "liet ke", "danh sach", "ke ten", "tim cac",
        "de xuat", "goi y", "gioi thieu", "cho toi xem", "co gi",
        "nhung san pham", "nhung thiet bi", "nhung loai",
        "san pham nao", "thiet bi nao", "loai nao", "tim giup",
        "cai nao", "dat nhat", "re nhat", "co ban", "tim mua", "o dau ban", "ben ban co"
    ]
    is_list_query = any(p in q_norm for p in list_patterns)
    
    advice_patterns = ["tu van", "muon mua", "can mua", "nen mua", "chon", "lua chon", "tim mua", "nen dung", "loi khuyen"]
    is_advice_query = any(p in q_norm for p in advice_patterns)
    
    has_company = bool(filters.get("company"))
    has_model = bool(filters.get("model_code"))
    has_type = bool(filters.get("product_type"))
    
    if is_advice_query and not has_company and not has_model:
        if has_type:
            return "general_advice", True
            
    if is_list_query:
        clarification = not (has_company or has_type)
        return "list", clarification
        
    if is_advice_query and not has_company and not has_model and not has_type:
        return "general_advice", True
        
    return "detail", False


# ============================================================
# GỌI API GEMINI AN TOÀN
# ============================================================

def format_history_for_gemini(history: list[ChatMessage]) -> list:
    """Chuyển đổi danh sách ChatMessage sang định dạng lịch sử cho Gemini SDK"""
    gemini_history = []
    for msg in history:
        role = "user" if msg.role == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg.content]})
    return gemini_history


def generate_content_safe(
    prompt: str, 
    system_instruction: str = None, 
    generation_config: dict = None, 
    history: list[ChatMessage] = None,
    max_attempts: int = 5
) -> str:
    """Gọi Gemini an toàn với Retry và xoay vòng API Key."""
    for attempt in range(max_attempts):
        api_key = key_manager.configure_sdk()
        if not api_key:
            return "Hệ thống chưa được cấu hình API Key của Gemini. Vui lòng cấu hình file `.env` ở backend."
            
        try:
            model = genai.GenerativeModel(
                model_name=gemini_model_name,
                system_instruction=system_instruction
            )
            
            if history:
                gemini_history = format_history_for_gemini(history)
                chat = model.start_chat(history=gemini_history)
                response = chat.send_message(prompt, generation_config=generation_config, request_options={"timeout": 60.0})
            else:
                response = model.generate_content(prompt, generation_config=generation_config, request_options={"timeout": 60.0})
                
            return response.text.strip()
        except (google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.DeadlineExceeded) as e:
            masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "Key"
            print(f"[Safe Call] Key {masked} (lượt {attempt + 1}) gặp lỗi: {e}")
            key_manager.mark_key_limit(api_key)
            if attempt == max_attempts - 1:
                raise e
        except Exception as e:
            if "time out" in str(e).lower() or "timeout" in str(e).lower() or "deadline" in str(e).lower():
                masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "Key"
                print(f"[Safe Call] Key {masked} (lượt {attempt + 1}) bị timeout: {e}")
                key_manager.mark_key_limit(api_key, cooldown_seconds=15.0)
                if attempt == max_attempts - 1:
                    raise e
            else:
                print(f"[Safe Call Error] Lỗi nghiêm trọng: {e}")
                raise e
            
            
    raise Exception("Đã thử tất cả API Keys nhưng đều gặp lỗi!")

def generate_content_stream_safe(
    prompt: str, 
    system_instruction: str = None, 
    generation_config: dict = None, 
    history: list[ChatMessage] = None,
    max_attempts: int = 5
):
    """Gọi Gemini an toàn với Retry và stream."""
    for attempt in range(max_attempts):
        api_key = key_manager.configure_sdk()
        if not api_key:
            yield {"type": "error", "content": "Hệ thống chưa được cấu hình API Key của Gemini. Vui lòng cấu hình file `.env` ở backend."}
            return
            
        try:
            model = genai.GenerativeModel(
                model_name=gemini_model_name,
                system_instruction=system_instruction
            )
            
            if history:
                gemini_history = format_history_for_gemini(history)
                chat = model.start_chat(history=gemini_history)
                response = chat.send_message(prompt, generation_config=generation_config, stream=True, request_options={"timeout": 60.0})
            else:
                response = model.generate_content(prompt, generation_config=generation_config, stream=True, request_options={"timeout": 60.0})
                
            for chunk in response:
                if chunk.text:
                    text = chunk.text
                    # Break the chunk into smaller pieces to create a smooth typing effect
                    step = 2
                    for i in range(0, len(text), step):
                        yield text[i:i+step]
                        time.sleep(0.01) # 10ms delay per 2 chars (~200 chars/s)
            return
        except (google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.DeadlineExceeded) as e:
            masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "Key"
            print(f"[Safe Call] Key {masked} (lượt {attempt + 1}) gặp lỗi: {e}")
            key_manager.mark_key_limit(api_key, cooldown_seconds=15.0)
            if attempt == max_attempts - 1:
                yield {"type": "error", "content": f"Lỗi sau nhiều lần thử: {e}"}
        except Exception as e:
            if "time out" in str(e).lower() or "timeout" in str(e).lower() or "deadline" in str(e).lower():
                masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "Key"
                print(f"[Safe Call] Key {masked} (lượt {attempt + 1}) bị timeout: {e}")
                key_manager.mark_key_limit(api_key, cooldown_seconds=15.0)
                if attempt == max_attempts - 1:
                    yield {"type": "error", "content": f"Lỗi timeout sau nhiều lần thử: {e}"}
            else:
                print(f"[Safe Call Error] Lỗi nghiêm trọng: {e}")
                yield {"type": "error", "content": f"Lỗi: {e}"}
                return
            
    yield {"type": "error", "content": "Đã thử tất cả API Keys nhưng đều gặp lỗi!"}

def single_chunk_stream(response_text: str, sources: list = None):
    def event_generator():
        yield json.dumps({"type": "chunk", "content": response_text}) + "\n"
        yield json.dumps({"type": "sources", "sources": sources or []}) + "\n"
    return StreamingResponse(event_generator(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

# ============================================================
# QUẢN LÝ LỊCH SỬ CHAT
# ============================================================

def condense_question(query: str, history: list[ChatMessage]) -> str:
    """Bổ sung ngữ cảnh cho câu hỏi dựa vào lịch sử chat bằng LLM."""
    if not history:
        return query
        
    # Lấy 4 tin nhắn gần nhất
    history_text = "\n".join([f"{'User' if m.role == 'user' else 'AI'}: {m.content}" for m in history[-4:]])
    
    prompt = (
        "Bạn là một chuyên gia phân tích ngữ nghĩa. Dưới đây là lịch sử trò chuyện ngắn và câu hỏi mới nhất của người dùng.\n"
        "Nhiệm vụ: Viết lại câu hỏi mới nhất sao cho nó đứng độc lập mà vẫn mang đầy đủ ngữ nghĩa để truy vấn.\n"
        "Quy tắc tối quan trọng:\n"
        "1. Nếu câu hỏi mới là một câu đính chính, mở rộng hoặc thu hẹp phạm vi của câu hỏi ngay trước đó (ví dụ: 'ý tôi là tất cả sản phẩm', 'tìm món rẻ nhất', 'loại màu đỏ'), BẮT BUỘC phải mang yêu cầu chính (intent) của câu hỏi trước đó ghép với phạm vi mới này.\n"
        "2. Nếu câu hỏi mới HỎI TIẾP về sản phẩm/hãng ở câu trước (dùng đại từ 'nó', 'loại này', 'giá bao nhiêu', ...), hãy thay thế đại từ bằng tên sản phẩm/hãng đó.\n"
        "3. Nếu câu hỏi mới ĐỔI CHỦ ĐỀ hoặc hỏi MỚI HOÀN TOÀN (ví dụ: 'có bán máy hút bụi không', 'công ty ở đâu'), CHỈ giữ lại nội dung của câu hỏi mới, TUYỆT ĐỐI KHÔNG mang bất kỳ yêu cầu cũ nào (như đắt nhất, rẻ nhất, thông số) từ câu trước sang.\n"
        "4. Nếu câu hỏi mới đã đầy đủ ngữ nghĩa, giữ nguyên.\n"
        "5. CHỈ trả về đúng MỘT CÂU được viết lại, không giải thích, không thêm ngoặc kép.\n\n"
        f"--- Lịch sử ---\n{history_text}\n"
        f"--- Câu hỏi mới ---\nUser: {query}\n"
        "--- Câu hỏi viết lại ---"
    )
    
    try:
        gemini_response = generate_content_safe(
            prompt=prompt,
            system_instruction=None,
            generation_config={"temperature": 0.0},
            max_attempts=3
        )
        rewritten_query = gemini_response.strip()
        print(f"[Query Rewrite] Gốc: '{query}' -> Mới: '{rewritten_query}'")
        return rewritten_query
    except Exception as e:
        print(f"[Query Rewrite Error] {e}. Fallback to original query.")
        return query


def generate_search_query(query: str) -> dict:
    """Phân tích câu hỏi thành cấu trúc truy vấn bằng heuristics."""
    meta = extract_metadata_from_text(query)
    action, clarification_needed = determine_action_heuristically(query, meta)
    
    result = {
        "action": action,
        "filters": {
            "company": meta["company"],
            "product_type": meta["product_type"],
            "model_code": meta["model_code"],
            "price_filter": meta.get("price_filter")
        },
        "search_query": query,
        "clarification_needed": clarification_needed
    }
    
    print(f"[Query Analysis] {result}")
    return result


# ============================================================
# TÌM KIẾM HYBRID
# ============================================================

def reciprocal_rank_fusion(ranked_lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion — Cormack et al. (2009)
    Kết hợp nhiều ranked list thành một ranking thống nhất.
    """
    fused_scores = {}
    for ranked_list in ranked_lists:
        for rank, (idx, _score) in enumerate(ranked_list):
            fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


def product_matches_brand_filter(prod: dict, brand_query: str) -> bool:
    """
    [GĐ1 FIX] Lọc hãng bằng OR logic: tìm trong CẢ brand, company VÀ device.
    Không còn phân biệt cứng ACTUAL_COMPANIES vs brands trong tên thiết bị.
    """
    if not brand_query:
        return True
    
    brand_norm = remove_vietnamese_accents(brand_query).strip().lower()
    
    # Kiểm tra trong trường brand (mới, trích xuất từ ingest)
    prod_brand_norm = remove_vietnamese_accents(prod.get("brand", ""))
    if brand_norm in prod_brand_norm:
        return True
    
    # Kiểm tra trong trường company (nhà phân phối)
    prod_company_norm = remove_vietnamese_accents(prod.get("company", ""))
    if brand_norm in prod_company_norm:
        return True
    
    # Kiểm tra trong tên thiết bị
    prod_device_norm = remove_vietnamese_accents(prod.get("device", ""))
    if brand_norm in prod_device_norm:
        return True
    
    return False


def product_matches_type_filter(prod: dict, type_keywords: list[str]) -> bool:
    """Lọc loại thiết bị dựa trên synonyms."""
    if not type_keywords:
        return True
    
    # Ưu tiên trường category (mới, trích xuất từ ingest)
    prod_category = prod.get("category", "").lower()
    prod_category_norm = remove_vietnamese_accents(prod_category)
    
    prod_device = prod.get("device", "").lower()
    prod_device_norm = remove_vietnamese_accents(prod_device)
    
    for kw in type_keywords:
        kw_norm = remove_vietnamese_accents(kw)
        # Tránh false positive: "kìm" matching "kim loại" hoặc "hợp kim"
        if kw_norm == "kim" and ("kim loai" in prod_device_norm or "hop kim" in prod_device_norm):
            continue
        if (kw in prod_category or kw_norm in prod_category_norm or
            kw in prod_device or kw_norm in prod_device_norm):
            return True
    
    return False


def weighted_reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]], 
    weights: list[float] = None,
    k: int = 60
) -> list[tuple[int, float]]:
    """
    Weighted RRF — mở rộng RRF chuẩn với trọng số cho từng ranked list.
    weights[i] càng cao → ranked_list[i] càng ảnh hưởng kết quả cuối.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    
    fused_scores = {}
    for w, ranked_list in zip(weights, ranked_lists):
        for rank, (idx, _score) in enumerate(ranked_list):
            fused_scores[idx] = fused_scores.get(idx, 0.0) + w / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


def build_reranker_text(prod: dict, query: str) -> str:
    """Tối ưu text cho Reranker: đẩy thông số kỹ thuật (specs) liên quan nhất lên đầu."""
    series = prod.get('series', '')
    specs = prod.get('specs', {})
    
    query_norm = remove_vietnamese_accents(query).lower()
    query_words = [w for w in re.findall(r'[a-z0-9]+', query_norm) if len(w) >= 2]
    
    relevant_specs = []
    other_specs = []
    
    if specs:
        for key, val in specs.items():
            key_norm = remove_vietnamese_accents(key).lower()
            val_norm = remove_vietnamese_accents(val).lower()
            is_relevant = any(w in key_norm or w in val_norm for w in query_words)
            if is_relevant:
                relevant_specs.append(f"{key}: {val}")
            else:
                other_specs.append(f"{key}: {val}")
                
    specs_text = " | ".join(relevant_specs + other_specs)[:300]
    desc_preview = prod.get('description', '')[:200]
    
    return f"{prod.get('brand', '')} {series} {prod.get('device', '')} | {specs_text} | {desc_preview}"[:512]


def hybrid_search(
    query: str, 
    query_vector: np.ndarray,
    brand_filter: str = None,
    type_filter: str = None,
    model_tokens: list[str] = None,
    price_filter: dict = None,
    top_k: int = 20
) -> list[tuple[dict, float]]:
    """
    Hybrid Search Pipeline (Cải tiến):
    1. Pre-filter bằng metadata (brand + category + series)
    2. Vector Search (cosine similarity)
    3. BM25 Keyword Search
    4. Weighted RRF — tăng trọng số BM25 khi query có mã kỹ thuật
    5. Cross-Encoder Reranking (nếu có)
    """
    global bm25_index, reranker_model
    
    # --- Phát hiện series tokens ---
    has_series_query = False
    series_tokens = []
    if model_tokens:
        for token in model_tokens:
            if is_broad_model_token(token):
                has_series_query = True
                series_tokens.append(token)
    
    # --- Bước 1: Pre-filter (Inverted Index O(1)) ---
    type_keywords = get_synonyms_for_type(type_filter) if (type_filter and not is_generic_type(type_filter)) else []
    
    candidate_set = None
    
    if brand_filter:
        brand_norm = remove_vietnamese_accents(brand_filter).strip().lower()
        b_candidates = set()
        for k, v in brand_index.items():
            if brand_norm in k:
                b_candidates.update(v)
        candidate_set = b_candidates if candidate_set is None else candidate_set.intersection(b_candidates)
        
    if type_keywords:
        t_candidates = set()
        for idx, prod in enumerate(metadata):
            if product_matches_type_filter(prod, type_keywords):
                t_candidates.add(idx)
        candidate_set = t_candidates if candidate_set is None else candidate_set.intersection(t_candidates)
        
    if model_tokens:
        m_candidates = set()
        for token in model_tokens:
            token_set = set()
            for k, v in sku_index.items():
                if token in k:
                    token_set.update(v)
            for k, v in series_index.items():
                if token == k:
                    token_set.update(v)
            # Fallback nếu model không nằm trong index (tên quá khác lạ)
            if not token_set:
                 for idx, prod in enumerate(metadata):
                      if token in normalize_sku(prod.get("device", "")):
                          token_set.add(idx)
            m_candidates.update(token_set)
        
        # model_tokens ghi đè type_filter (vì người dùng gọi chính xác model) nhưng vẫn giữ brand filter nếu có
        candidate_set = m_candidates
        if brand_filter:
            brand_norm = remove_vietnamese_accents(brand_filter).strip().lower()
            b_candidates = set()
            for k, v in brand_index.items():
                if brand_norm in k:
                    b_candidates.update(v)
            candidate_set = candidate_set.intersection(b_candidates)
            
    if candidate_set is None or len(candidate_set) == 0:
        print("[Search] Pre-filter bằng Inverted Index trả 0 kết quả hoặc không có filter. Fallback toàn bộ database.")
        candidate_indices = list(range(len(metadata)))
    else:
        candidate_indices = list(candidate_set)
        
    # --- Bước 1.5: Lọc bằng giá (nếu có) ---
    if price_filter:
        min_p = price_filter.get("min")
        max_p = price_filter.get("max")
        filtered_indices = []
        for idx in candidate_indices:
            prod_price = metadata[idx].get("price")
            if prod_price is not None:
                if min_p is not None and prod_price < min_p:
                    continue
                if max_p is not None and prod_price > max_p:
                    continue
                filtered_indices.append(idx)
        candidate_indices = filtered_indices
        print(f"[Search] Candidates sau khi lọc giá: {len(candidate_indices)}")
    
    print(f"[Search] Candidates sau pre-filter: {len(candidate_indices)} (series_query={has_series_query})")
    
    # --- Bước 2: Vector Search bằng Qdrant ---
    global qdrant_client_instance
    
    qdrant_filter = None
    if candidate_indices and len(candidate_indices) < len(metadata):
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="product_idx",
                    match=MatchAny(any=candidate_indices)
                )
            ]
        )
        
    try:
        search_result = qdrant_client_instance.query_points(
            collection_name="products",
            query=query_vector.tolist(),
            query_filter=qdrant_filter,
            limit=top_k * 6  # Lấy dư để deduplicate
        ).points
        
        # Deduplicate (Qdrant trả về multiple chunks của cùng 1 product)
        product_max_scores = {}
        for hit in search_result:
            prod_idx = hit.payload.get("product_idx")
            if prod_idx is not None:
                if prod_idx not in product_max_scores or hit.score > product_max_scores[prod_idx]:
                    product_max_scores[prod_idx] = hit.score
                    
        vector_ranked = sorted(product_max_scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 3]
    except Exception as e:
        print(f"[Search Error] Lỗi khi gọi Qdrant: {e}")
        vector_ranked = []
    
    # --- Bước 3: BM25 Search trên candidates ---
    query_tokens = tokenize_vietnamese(query)
    bm25_scores_all = bm25_index.get_scores(query_tokens)
    bm25_scores = [(idx, float(bm25_scores_all[idx])) for idx in candidate_indices]
    bm25_ranked = sorted(bm25_scores, key=lambda x: x[1], reverse=True)[:top_k * 3]
    
    # --- Bước 4: Weighted RRF ---
    # Khi query có mã kỹ thuật/SKU → tăng trọng số BM25 (match chính xác quan trọng hơn semantic)
    has_sku = bool(model_tokens)
    if has_sku:
        rrf_weights = [0.7, 1.3]  # [vector, bm25] — ưu tiên BM25 cho SKU
    else:
        rrf_weights = [1.0, 1.0]  # Equal weight cho query tự nhiên
    
    fused = weighted_reciprocal_rank_fusion([vector_ranked, bm25_ranked], weights=rrf_weights, k=60)
    
    # Boost chính xác cho SKU match
    if model_tokens:
        boosted = []
        for idx, score in fused:
            prod_sku = normalize_sku(metadata[idx].get("device", ""))
            for token in model_tokens:
                if token and token in prod_sku:
                    score += 0.05
                    # Exact match bonus
                    if prod_sku == token or prod_sku.startswith(token):
                        score += 0.03
            boosted.append((idx, score))
        fused = sorted(boosted, key=lambda x: x[1], reverse=True)
    
    top_candidates = fused[:top_k]
    
    # --- Bước 5: Cross-Encoder Reranking ---
    if reranker_model and len(top_candidates) > 1:
        rerank_limit = min(6, len(top_candidates))
        rerank_candidates = top_candidates[:rerank_limit]
        pairs = []
        for idx, _score in rerank_candidates:
            prod = metadata[idx]
            prod_text = build_reranker_text(prod, query)
            pairs.append([query, prod_text])
        
        try:
            rerank_scores = reranker_model.predict(pairs)
            reranked = sorted(
                zip([idx for idx, _ in rerank_candidates], rerank_scores),
                key=lambda x: x[1], reverse=True
            )
            print(f"[Reranker] Reranked {len(reranked)} candidates.")
            reranked_results = [(metadata[idx], float(score)) for idx, score in reranked]
            remaining_results = [(metadata[idx], score) for idx, score in top_candidates[rerank_limit:]]
            return reranked_results + remaining_results
        except Exception as e:
            print(f"[Reranker] Lỗi reranking: {e}. Dùng kết quả RRF.")
    
    return [(metadata[idx], score) for idx, score in top_candidates]


# ============================================================
# LLM QUERY EXPANSION & PROMPT ENGINEERING
# ============================================================

def expand_query_with_llm(query: str, filters: dict, history: list[ChatMessage] = None) -> str:
    """Sử dụng LLM để mở rộng từ khóa tìm kiếm khi query quá chung chung."""
    if not key_manager.api_keys:
        return query
        
    comp_filter = filters.get("company")
    type_filter = filters.get("product_type")
    
    prompt = (
        f"Câu hỏi của người dùng: '{query}'\n"
        f"Hãng: {comp_filter or 'Chưa rõ'}\n"
        f"Loại thiết bị: {type_filter or 'Chưa rõ'}\n"
        "Nhiệm vụ: Trích xuất các từ khóa đặc tả chuyên ngành từ câu hỏi trên để tối ưu hóa tìm kiếm. "
        "Chỉ trả về chuỗi từ khóa, KHÔNG giải thích."
    )
    
    try:
        expanded = generate_content_safe(prompt, generation_config={"temperature": 0.0})
        if len(expanded) > 0 and len(expanded) < 100:
            print(f"[Query Expansion] '{query}' -> '{expanded}'")
            return f"{query} {expanded}"
    except Exception as e:
        print(f"[Query Expansion Error] {e}")
        
    return query

BASE_SYSTEM_PROMPT = """Bạn là một trợ lý ảo tư vấn thiết bị công nghiệp thông minh của công ty Đại Dương Automation.
NGUYÊN TẮC QUAN TRỌNG NHẤT: TUYỆT ĐỐI KHÔNG BỊA ĐẶT THÔNG SỐ. 
1. CHỈ dùng thông tin trong <products_context> hoặc dữ liệu thống kê. Không bịa giá bán hoặc thông số.
2. Ưu tiên trích dẫn dữ liệu từ thẻ <evidence_specs> vì đây là các thông số quan trọng nhất.
3. Hiển thị rõ Hãng sản xuất (brand) và Nhà phân phối (distributor).
4. Nếu dữ liệu không có hãng người dùng hỏi, hãy trả lời rõ là chưa có hãng đó trong hệ thống.
5. Định dạng markdown đẹp, in đậm tên sản phẩm. TUYỆT ĐỐI KHÔNG thụt lề bằng dấu cách (space) ở đầu dòng để tránh lỗi giao diện hiển thị thanh cuộn ngang.
"""

# ============================================================
# MAIN CHAT ENDPOINT
# ============================================================

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    global qdrant_client_instance, metadata, embedding_model, query_embedding_cache, search_result_cache
    
    if qdrant_client_instance is None or metadata is None or embedding_model is None:
        init_backend()
        if qdrant_client_instance is None:
            raise HTTPException(status_code=500, detail="Qdrant DB not initialized. Please ensure Docker is running and run ingest.py first.")
            
    query = request.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    try:
        # BƯỚC 1: Chat Memory
        condensed_query = condense_question(query, request.history)
        
        # BƯỚC 2: Query Analysis
        query_data = generate_search_query(condensed_query)
        
        action = query_data.get("action", "detail")
        filters = query_data.get("filters", {})
        search_query = query_data.get("search_query") or condensed_query
        clarification_needed = query_data.get("clarification_needed", False)
        
        comp_filter = filters.get("company")
        type_filter = filters.get("product_type")
        model_filter = filters.get("model_code")
        
        print(f"[Chat] action={action}, company={comp_filter}, type={type_filter}, models={model_filter}, clarification={clarification_needed}")
        
        # NHÁNH CHITCHAT
        if action == "chitchat":
            system_prompt = (
                "Bạn là một trợ lý ảo tư vấn thiết bị công nghiệp thông minh, nhiệt tình và am hiểu sản phẩm "
                "của công ty Đại Dương Automation. "
                "Hãy trả lời câu hỏi giao tiếp thông thường, chào hỏi, hoặc giới thiệu bản thân một cách thân thiện, ngắn gọn và lịch sự bằng tiếng Việt."
            )
            if not key_manager.api_keys:
                return single_chunk_stream("Xin chào! Hiện tại hệ thống chưa cấu hình Gemini API Key. Vui lòng cấu hình file .env.", [])
                
            def chitchat_generator():
                try:
                    stream_gen = generate_content_stream_safe(prompt=query, system_instruction=system_prompt, generation_config={"temperature": 0.7})
                    for chunk_text in stream_gen:
                        if isinstance(chunk_text, dict) and chunk_text.get("type") == "error":
                            yield json.dumps(chunk_text) + "\n"
                            break
                        else:
                            yield json.dumps({"type": "chunk", "content": chunk_text}) + "\n"
                except Exception as e:
                    print(f"[Chitchat Error] {e}")
                    yield json.dumps({"type": "chunk", "content": "Xin chào! Rất vui được hỗ trợ bạn. Tôi sẵn sàng tư vấn các thiết bị công nghiệp cho bạn."}) + "\n"
                
                yield json.dumps({"type": "sources", "sources": []}) + "\n"
                
            return StreamingResponse(chitchat_generator(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

        # BƯỚC 3: Quét hãng có sẵn theo loại thiết bị (phục vụ clarification)
        available_brands = []
        if type_filter and not is_generic_type(type_filter):
            type_kws = get_synonyms_for_type(type_filter)
            brands_found = set()
            for prod in metadata:
                if product_matches_type_filter(prod, type_kws):
                    brands_found.add(prod.get("brand", prod.get("company")))
            available_brands = sorted(list(brands_found))

        # NHÁNH GENERAL ADVICE / CLARIFICATION
        if clarification_needed or action == "general_advice":
            context_brands = f"Loại thiết bị khách hỏi: {type_filter if type_filter else 'Chưa rõ'}\n"
            if available_brands:
                context_brands += f"Các hãng sản xuất trong hệ thống đang cung cấp loại sản phẩm này:\n"
                for brand in available_brands:
                    context_brands += f"  - {brand}\n"
            else:
                context_brands += "Không tìm thấy hãng cụ thể nào trong dữ liệu cung cấp loại này.\n"
                
            # Lấy danh sách TẤT CẢ danh mục có thực trong DB để LLM gợi ý (không tự bịa)
            all_categories = sorted(list(set(prod.get("category", "") for prod in metadata if prod.get("category"))))
            if not type_filter or type_filter == "all" or is_generic_type(type_filter):
                context_brands += "\nDanh sách các LOẠI THIẾT BỊ HIỆN CÓ trong hệ thống để bạn gợi ý cho khách:\n"
                for cat in all_categories:
                    context_brands += f"  - {cat}\n"
                
            system_prompt = (
                "Bạn là một trợ lý ảo tư vấn thiết bị công nghiệp thông minh của Đại Dương Automation.\n"
                "Khách hàng đang yêu cầu tư vấn chung chung về một loại thiết bị nhưng chưa chỉ rõ hãng sản xuất hoặc model cụ thể.\n"
                "Nhiệm vụ của bạn:\n"
                "1. Hãy trả lời lịch sự, giới thiệu các hãng chúng tôi đang cung cấp dựa trên dữ liệu bên dưới.\n"
                "2. Hỏi ngược lại khéo léo để khách chọn hãng sản xuất hoặc chia sẻ thêm nhu cầu cụ thể.\n"
                "3. Nếu không rõ loại thiết bị, hãy GỢI Ý các loại thiết bị dựa vào 'LOẠI THIẾT BỊ HIỆN CÓ' bên dưới. TUYỆT ĐỐI KHÔNG tự bịa ra (như Robot công nghiệp v.v.) nếu nó không có trong danh sách.\n"
                "4. Nếu có nhiều hãng, hãy giới thiệu sơ lược ưu điểm của 2-3 hãng tiêu biểu.\n"
                "5. Định dạng câu trả lời đẹp bằng markdown, viết bằng tiếng Việt.\n\n"
                f"--- Dữ liệu hiện có ---\n{context_brands}"
            )
            
            if not key_manager.api_keys:
                brand_list = ', '.join(available_brands[:5]) if available_brands else 'OMRON, TOHO Electronics, Oriental Motor'
                return single_chunk_stream(f"Chúng tôi cung cấp thiết bị của nhiều thương hiệu lớn. Đối với nhóm **{type_filter if type_filter else 'sản phẩm này'}**, bạn có thể tham khảo thiết bị của các hãng như {brand_list}. Bạn quan tâm tới hãng nào?", [])
                
            def general_advice_generator():
                try:
                    stream_gen = generate_content_stream_safe(prompt=query, system_instruction=system_prompt, generation_config={"temperature": 0.7})
                    for chunk_text in stream_gen:
                        if isinstance(chunk_text, dict) and chunk_text.get("type") == "error":
                            yield json.dumps(chunk_text) + "\n"
                            break
                        else:
                            yield json.dumps({"type": "chunk", "content": chunk_text}) + "\n"
                except Exception as e:
                    print(f"[General Advice Error] {e}")
                    brand_list = ', '.join(available_brands[:5]) if available_brands else 'OMRON, TOHO Electronics'
                    fallback = f"Chúng tôi cung cấp thiết bị của nhiều thương hiệu lớn. Đối với nhóm **{type_filter if type_filter else 'sản phẩm này'}**, bạn có thể tham khảo thiết bị của các hãng như {brand_list}. Bạn quan tâm tới hãng nào?"
                    yield json.dumps({"type": "chunk", "content": fallback}) + "\n"
                
                yield json.dumps({"type": "sources", "sources": []}) + "\n"
                
            return StreamingResponse(general_advice_generator(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

        # BƯỚC 3.5: LLM Query Expansion (chỉ chạy khi không có model cụ thể)
        if not model_filter and not action == "chitchat":
            search_query = expand_query_with_llm(search_query, filters, request.history)

        # BƯỚC 4: Hybrid Search Pipeline (Vector + BM25 + RRF + Reranker)
        
        # --- SEMANTIC CACHING ---
        cache_key = f"{search_query}_{comp_filter}_{type_filter}_{str(model_filter)}"
        now = time.time()
        
        # Dọn dẹp cache cũ (> 5 phút)
        expired = [k for k, v in search_result_cache.items() if now - v['time'] > 300]
        for k in expired: del search_result_cache[k]
            
        if cache_key in search_result_cache:
            matched_products = search_result_cache[cache_key]['results']
            matched_count = len(matched_products)
            print(f"[Cache Hit] Lấy {matched_count} sản phẩm từ cache.")
        else:
            if search_query in query_embedding_cache:
                query_vector = query_embedding_cache[search_query]
            else:
                query_vector = embedding_model.encode(f"query: {search_query}", convert_to_numpy=True)
                norm = np.linalg.norm(query_vector)
                if norm > 0: query_vector = query_vector / norm
                if len(query_embedding_cache) > 1000: query_embedding_cache.clear()
                query_embedding_cache[search_query] = query_vector

            search_results = hybrid_search(
                query=search_query,
                query_vector=query_vector,
                brand_filter=comp_filter,
                type_filter=type_filter,
                model_tokens=model_filter,
                price_filter=filters.get("price_filter"),
                top_k=20
            )
            matched_products = [prod for prod, _score in search_results]
            matched_count = len(matched_products)
            search_result_cache[cache_key] = {'time': now, 'results': matched_products}
            print(f"[Search] Tìm thấy {matched_count} sản phẩm phù hợp.")

        # BƯỚC 5: Sinh System Prompt theo Action
        
        # --- NHÁNH COUNT ---
        if action == "count":
            # Đếm chính xác bằng pre-filter (không dùng kết quả search top-K)
            type_kws = get_synonyms_for_type(type_filter) if (type_filter and not is_generic_type(type_filter)) else []
            exact_count = 0
            
            p_filter = filters.get("price_filter")
            
            for prod in metadata:
                if product_matches_brand_filter(prod, comp_filter) and product_matches_type_filter(prod, type_kws):
                    if p_filter:
                        prod_price = prod.get("price")
                        if prod_price is None:
                            continue
                        if p_filter.get("min") is not None and prod_price < p_filter["min"]:
                            continue
                        if p_filter.get("max") is not None and prod_price > p_filter["max"]:
                            continue
                    exact_count += 1
            
            sample_products = matched_products[:5]
            display_company = comp_filter.upper() if comp_filter else ""
            display_type = type_filter if type_filter else ""
                
            context_info = (
                f"=== THỐNG KÊ THỰC TẾ TỪ CƠ SỞ DỮ LIỆU ===\n"
                f"- TỔNG SỐ LƯỢNG SẢN PHẨM KHỚP: {exact_count} sản phẩm\n"
            )
            if display_company:
                context_info += f"- Hãng/Thương hiệu: {display_company}\n"
            if display_type:
                context_info += f"- Loại thiết bị: {display_type}\n"
                
            if sample_products:
                context_info += (
                    f"\n=== DANH SÁCH MẪU TIÊU BIỂU (Tối đa 5 mẫu, KHÔNG phải toàn bộ {exact_count} sản phẩm) ===\n"
                    + "\n".join([f"- **{p['device']}** (Hãng {p.get('brand', p['company'])}, giá: {p['price_formatted']})" for p in sample_products])
                )
            
            system_prompt = BASE_SYSTEM_PROMPT + (
                "\nNhiệm vụ: Sử dụng dữ liệu thống kê thực tế để trả lời câu hỏi.\n"
                f"CHÚ Ý: Bạn PHẢI sử dụng con số '{exact_count}' sản phẩm để trả lời. TUYỆT ĐỐI KHÔNG tự đếm số dòng trong danh sách mẫu.\n"
                f"--- Dữ liệu thống kê ---\n{context_info}"
            )
            
        # --- NHÁNH COMPARE ---
        elif action == "compare":
            compare_limit = 6  # Nạp top 6 sản phẩm vào prompt để tránh bị "trôi" mất các mẫu nếu 1 model có quá nhiều biến thể
            sample_products = matched_products[:compare_limit]
            context_parts = []
            for i, prod in enumerate(sample_products, 1):
                evidence_xml = build_evidence_xml(prod, search_query, mode="compare")
                context_parts.append(f"<product id=\"{i}\">\n{evidence_xml}\n</product>")
            context_text = "\n".join(context_parts) if context_parts else "Không có sản phẩm so sánh phù hợp."
            
            system_prompt = BASE_SYSTEM_PROMPT + (
                "\nDưới đây là thông tin các sản phẩm cần so sánh trong các thẻ XML:\n"
                f"<products_context>\n{context_text}\n</products_context>\n"
                "Nhiệm vụ: Lập BẢN SO SÁNH BẰNG VĂN BẢN (Text) đối chiếu các tiêu chí cốt lõi.\n"
                "TUYỆT ĐỐI KHÔNG DÙNG BẢNG (Markdown Table). Hãy trình bày thành các đoạn văn ngắn hoặc gạch đầu dòng rõ ràng.\n\n"
                "Ví dụ định dạng:\n"
                "**1. Tiêu chí A:**\n"
                "- SP 1: 10mm\n"
                "- SP 2: 20mm\n\n"
                "**2. Tiêu chí B:**\n"
                "- SP 1: Tính năng X\n"
                "- SP 2: Tính năng Y\n\n"
                "Nếu CÙNG LOẠI: so sánh chi tiết thông số và đưa lời khuyên. Nếu KHÁC LOẠI: làm rõ đặc điểm riêng, KHÔNG so sánh ưu/nhược."
            )
            
        # --- NHÁNH LIST ---
        elif action == "list":
            sample_products = matched_products[:4]
            # Đếm chính xác
            type_kws = get_synonyms_for_type(type_filter) if (type_filter and not is_generic_type(type_filter)) else []
            total_found = sum(1 for prod in metadata 
                           if product_matches_brand_filter(prod, comp_filter) and product_matches_type_filter(prod, type_kws))
            if total_found == 0:
                total_found = matched_count
            
            context_parts = []
            for i, prod in enumerate(sample_products, 1):
                evidence_xml = build_evidence_xml(prod, search_query, mode="list")
                context_parts.append(f"<product id=\"{i}\">\n{evidence_xml}\n</product>")
            context_text = "\n".join(context_parts) if context_parts else "Không tìm thấy sản phẩm phù hợp."
            
            system_prompt = BASE_SYSTEM_PROMPT + (
                f"\nKhách hàng muốn xem danh sách sản phẩm. Tổng số sản phẩm tìm được: {total_found}.\n"
                "Dưới đây là danh sách sản phẩm tiêu biểu trong các thẻ XML:\n"
                f"<products_context>\n{context_text}\n</products_context>\n"
                f"Nhiệm vụ: Giới thiệu sản phẩm mạch lạc, gọn gàng. Nêu rõ tổng số sản phẩm tìm được là {total_found} (danh sách trên chỉ hiển thị tiêu biểu).\n"
                "Gợi ý khách hàng hỏi thêm chi tiết về sản phẩm cụ thể."
            )
            
        # --- NHÁNH DETAIL ---
        else:
            sample_products = matched_products[:3]
            context_parts = []
            for i, prod in enumerate(sample_products, 1):
                evidence_xml = build_evidence_xml(prod, search_query, mode="detail")
                context_parts.append(f"<product id=\"{i}\">\n{evidence_xml}\n</product>")
            context_text = "\n".join(context_parts) if context_parts else "Không có sản phẩm nào phù hợp trong cơ sở dữ liệu."
            system_prompt = BASE_SYSTEM_PROMPT + (
                "\nDưới đây là thông tin các sản phẩm liên quan trong các thẻ XML:\n"
                f"<products_context>\n{context_text}\n</products_context>\n"
                "Nhiệm vụ: Trả lời chi tiết, mạch lạc, lịch sự, tập trung vào nội dung khách hỏi.\n"
                "Nếu không tìm thấy thông tin phù hợp, hãy lịch sự thông báo và đề nghị liên hệ hotline. Chỉ nhắc giá bán khi khách hỏi giá hoặc đang tư vấn mua hàng."
            )
        
        # BƯỚC 6: Gọi Gemini sinh câu trả lời
        source_limit = max(1, min(request.top_k or 6, 10))
        return_sources = matched_products[:source_limit]
        
        def event_generator():
            try:
                if not key_manager.api_keys:
                    yield json.dumps({"type": "error", "content": "Hệ thống chưa được cấu hình API Key của Gemini. Vui lòng cấu hình file `.env` ở backend."}) + "\n"
                    yield json.dumps({"type": "sources", "sources": return_sources}) + "\n"
                    return
                
                stream_gen = generate_content_stream_safe(
                    prompt=query,
                    system_instruction=system_prompt,
                    generation_config={"temperature": 0.0},
                    history=request.history
                )
                
                for chunk_text in stream_gen:
                    if isinstance(chunk_text, dict) and chunk_text.get("type") == "error":
                        yield json.dumps(chunk_text) + "\n"
                        break
                    else:
                        yield json.dumps({"type": "chunk", "content": chunk_text}) + "\n"
                        
                # Send sources at the very end
                yield json.dumps({"type": "sources", "sources": return_sources}) + "\n"
                        
            except Exception as e:
                print(f"[Final Generation Error] {e}")
                yield json.dumps({"type": "error", "content": f"Gặp lỗi khi gọi Gemini API: {str(e)}"}) + "\n"
                yield json.dumps({"type": "sources", "sources": return_sources}) + "\n"

        return StreamingResponse(event_generator(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    env_mode = os.environ.get("ENV_MODE", "production")
    is_reload = True if env_mode == "development" else False
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=is_reload)