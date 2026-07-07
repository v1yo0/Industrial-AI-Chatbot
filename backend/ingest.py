import os
import sys
import json
import re
import unicodedata
from pathlib import Path

# Đảm bảo mã hóa UTF-8 cho console trên Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.models import VectorParams, Distance, PointStruct


# ============================================================
# BRAND & CATEGORY EXTRACTION (Contextual Retrieval)
# ============================================================

def remove_vietnamese_accents(input_str: str) -> str:
    """Loại bỏ dấu tiếng Việt để so khớp dễ dàng hơn."""
    if not input_str:
        return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return only_ascii.replace('đ', 'd').replace('Đ', 'D').lower()


# Danh sách hãng sản xuất nhận dạng — sắp xếp dài trước để match chính xác
KNOWN_BRANDS = sorted([
    "TOHO Electronics", "Oriental Motor", "Sanyo Denki", "Weidmuller",
    "Yodogawa", "KOMAX", "OMRON", "JUMO", "Huba Control",
    "Master Flex", "MasterFlex", "Balluff", "Endress Hauser", "Endress+Hauser",
    "Keyence", "Mitsubishi", "Siemens", "Schneider", "ABB",
    "Panasonic", "Fuji Electric", "Autonics", "IDEC", "Delta",
    "SICK", "IFM", "Turck", "Pepperl Fuchs", "Banner",
    "Azbil", "Yokogawa", "Honeywell", "Phoenix Contact",
    "Mira", "Cosmic", "Daikin", "SMC", "CKD", "Festo",
    "NB", "AS ONE", "Renishaw", "Nachi", "NSK", "THK",
    "Hiwin", "IAI", "Yaskawa", "Fanuc", "Beckhoff",
], key=len, reverse=True)


# Từ điển phân loại thiết bị — sắp xếp dài trước
CATEGORY_RULES = [
    # Cảm biến — phân loại chi tiết
    ("cảm biến tiệm cận", ["cảm biến tiệm cận", "cam bien tiem can", "proximity sensor", "proximity"]),
    ("cảm biến quang", ["cảm biến quang", "cam bien quang", "photoelectric", "photo sensor"]),
    ("cảm biến áp suất", ["cảm biến áp suất", "cam bien ap suat", "pressure sensor", "pressure transmitter"]),
    ("cảm biến nhiệt độ", ["cảm biến nhiệt", "cam bien nhiet", "can nhiệt", "can nhiet", "cặp nhiệt", "thermocouple", "pt100", "rtd"]),
    ("cảm biến khoảng cách", ["cảm biến khoảng cách", "cam bien khoang cach", "distance sensor"]),
    ("cảm biến dịch chuyển", ["cảm biến dịch chuyển", "cảm biến độ dịch", "displacement sensor"]),
    ("cảm biến", ["cảm biến", "cam bien", "sensor"]),
    # Bộ điều khiển
    ("bộ điều khiển nhiệt độ", ["bộ điều khiển nhiệt độ", "bo dieu khien nhiet do", "temperature controller", "đồng hồ nhiệt độ", "dong ho nhiet do"]),
    ("bộ điều khiển", ["bộ điều khiển", "bo dieu khien", "controller", "driver điều khiển"]),
    # Động cơ
    ("động cơ", ["động cơ", "dong co", "motor", "servo motor", "stepping motor", "geared motor"]),
    # Máy móc
    ("máy ghi dữ liệu", ["máy ghi", "may ghi", "recorder", "bộ ghi dữ liệu"]),
    ("máy sấy", ["máy sấy", "may say", "dryer"]),
    ("máy tuốt dây", ["máy tuốt", "may tuot"]),
    ("máy hút bụi", ["máy hút bụi", "may hut bui", "vacuum"]),
    ("máy cắt", ["máy cắt", "may cat", "cutter", "air shears", "air heat cutter"]),
    # Thiết bị khác
    ("bộ chuyển đổi tín hiệu", ["bộ chuyển đổi tín hiệu", "signal converter", "signal isolator"]),
    ("van", ["van điện từ", "van", "valve", "solenoid valve"]),
    ("xi lanh", ["xi lanh", "cylinder"]),
    ("khúc xạ kế", ["khúc xạ kế", "khuc xa ke", "refractometer"]),
    ("kìm", ["kìm", "kim", "plier", "crimping"]),
    ("găng tay", ["găng tay", "gang tay", "glove"]),
    ("quạt", ["quạt", "quat", "fan"]),
    ("hộp số", ["hộp số", "hop so", "gearbox", "geared"]),
    ("bơm", ["bơm", "bom", "pump", "ống bơm"]),
    ("đầu nối", ["đầu nối", "dau noi", "connector", "terminal"]),
    ("biến tần", ["biến tần", "bien tan", "inverter"]),
    ("plc", ["plc", "bộ lập trình"]),
    ("relay", ["relay", "rơ le", "ro le"]),
    ("nguồn", ["nguồn cấp", "nguon cap", "power supply"]),
    ("kích thủy lực", ["kích thủy lực", "kich thuy luc", "hydraulic jack", "con rùa"]),
    ("ống", ["ống", "hose", "tube", "curl hose"]),
    ("màn hình", ["màn hình", "man hinh", "display", "hmi"]),
    ("tủ điện", ["tủ điện", "tu dien", "tủ điều khiển", "panel"]),
    ("dây cáp", ["dây cáp", "day cap", "cable", "dây điện"]),
]


def extract_brand(device_name: str, company: str) -> str:
    """
    Trích xuất hãng sản xuất thực tế từ tên thiết bị.
    Ưu tiên match hãng dài nhất trước.
    Nếu không tìm thấy, fallback về company.
    """
    if not device_name:
        return company or "Không rõ"
    
    device_lower = device_name.lower()
    device_norm = remove_vietnamese_accents(device_name)
    
    for brand in KNOWN_BRANDS:
        brand_lower = brand.lower()
        brand_norm = remove_vietnamese_accents(brand)
        
        if brand_lower in device_lower or brand_norm in device_norm:
            return brand
    
    # Nếu không tìm thấy brand trong device, dùng company
    return company or "Không rõ"


def extract_category(device_name: str, description: str = "") -> str:
    """
    Phân loại thiết bị tự động dựa trên tên và mô tả.
    Ưu tiên match cụ thể trước, chung sau.
    """
    if not device_name:
        return "khác"
    
    combined = (device_name + " " + (description[:500] if description else "")).lower()
    combined_norm = remove_vietnamese_accents(combined)
    
    for category, keywords in CATEGORY_RULES:
        for kw in keywords:
            kw_norm = remove_vietnamese_accents(kw)
            if kw in combined or kw_norm in combined_norm:
                return category
    
    return "khác"


def extract_series(device_name: str) -> str:
    """
    Trích xuất mã series/dòng sản phẩm từ tên thiết bị.
    VD: 'E5CC-QX3ASM-001' → 'E5CC'
        'ACT20C-GTI-LOOP-S' → 'ACT20C'
        'TM-T4Y' → 'TM'
    """
    if not device_name:
        return ""
    
    # Tách tên thiết bị: bỏ phần brand prefix nếu có
    # Lấy các token alphanumeric
    tokens = re.findall(r'[A-Za-z0-9][-A-Za-z0-9]*', device_name)
    
    # Đơn vị đo & suffix cần loại bỏ (false positive)
    unit_suffixes = {'mm', 'cm', 'm', 'km', 'kw', 'w', 'vac', 'vdc', 'v', 'a', 'ma', 'kg', 'g', 'mpa', 'bar', 'psi', 'hz', 'mhz', 'ghz', 'ml', 'cc', 'ohm'}
    
    for token in tokens:
        # Bỏ qua token chỉ là brand name đã biết
        token_lower = token.lower()
        if any(token_lower == remove_vietnamese_accents(b).strip() for b in KNOWN_BRANDS):
            continue
        
        # Series = phần trước dấu '-' đầu tiên, có cả chữ lẫn số, >= 2 ký tự
        parts = token.split('-')
        candidate = parts[0]
        has_digit = any(c.isdigit() for c in candidate)
        has_alpha = any(c.isalpha() for c in candidate)
        
        if len(candidate) >= 2 and has_digit and has_alpha:
            # Lọc false positive: bỏ token dạng "500MM", "75KW", "10M" (số + đơn vị)
            candidate_upper = candidate.upper()
            # Tách phần chữ cuối của candidate
            alpha_suffix = re.sub(r'^[0-9]+', '', candidate_upper).lower()
            if alpha_suffix in unit_suffixes:
                continue  # Đây là giá trị đo, không phải series
            
            # Series thực thường bắt đầu bằng chữ cái (E5CC, ACT20C, CD33...)
            if candidate[0].isalpha():
                return candidate_upper
        
        # Nếu token không có dấu '-' nhưng bản thân nó là mã sản phẩm
        if len(parts) == 1 and len(candidate) >= 3 and has_digit and has_alpha:
            alpha_suffix = re.sub(r'^[0-9]+', '', candidate.upper()).lower()
            if alpha_suffix in unit_suffixes:
                continue
            if candidate[0].isalpha():
                return candidate.upper()
    
    return ""


def extract_structured_specs(description: str) -> dict:
    """
    Parse mô tả sản phẩm thành dict key-value structured.
    Giúp BM25 match chính xác trên thông số kỹ thuật.
    """
    if not description:
        return {}
    
    desc_str = str(description).strip()
    specs = {}
    
    # Tách theo | hoặc \n
    lines = []
    if "|" in desc_str:
        lines = [p.strip() for p in desc_str.split("|") if p.strip()]
    elif "\n" in desc_str:
        lines = [p.strip() for p in desc_str.split("\n") if p.strip()]
    else:
        lines = [desc_str]
    
    for line in lines:
        # Tìm pattern key: value hoặc key： value
        for sep in [':', '：']:
            if sep in line:
                key, val = line.split(sep, 1)
                key = key.strip().upper()
                val = val.strip()
                if key and val and len(key) <= 60:
                    specs[key] = val
                break
    
    return specs


def specs_to_search_text(specs: dict) -> str:
    """
    Chuyển specs dict thành chuỗi text ngắn gọn cho BM25 indexing.
    Chỉ giữ các value chứa thông số kỹ thuật quan trọng.
    """
    if not specs:
        return ""
    
    # Các key quan trọng cho search
    priority_keys = {
        "NGUỒN CẤP", "NGUON CAP", "ĐIỆN ÁP", "DIEN AP", "PHẠM VI ĐIỆN ÁP HOẠT ĐỘNG",
        "ĐẦU VÀO CẢM BIẾN", "DAU VAO CAM BIEN", "ĐẦU VÀO CẢM BIẾN NHIỆT ĐỘ",
        "LOẠI ĐẦU RA", "LOAI DAU RA", "ĐẦU RA", "DAU RA", "NGÕ RA",
        "PHƯƠNG PHÁP CẢM BIẾN", "KHOẢNG CÁCH CẢM BIẾN",
        "CÔNG SUẤT", "CONG SUAT", "CÔNG SUẤT TIÊU THỤ",
        "TÊN SẢN PHẨM", "MÃ HÀNG", "DÒNG SẢN PHẨM", "DÒNG",
    }
    
    parts = []
    for key, val in specs.items():
        key_upper = remove_vietnamese_accents(key).upper()
        # Thêm tất cả specs nhưng ưu tiên các key quan trọng
        if any(remove_vietnamese_accents(pk).upper() in key_upper for pk in priority_keys):
            parts.insert(0, f"{key}: {val}")
        else:
            parts.append(val)  # Chỉ lấy value cho các key phụ
    
    return " | ".join(parts[:10])  # Giới hạn 10 specs để tránh loãng


def create_product_chunks(prod: dict) -> list[dict]:
    """
    Tạo multiple chunks cho mỗi sản phẩm để embedding chính xác hơn.
    - Chunk "identity": Tên, hãng, series, loại, giá (LUÔN có)
    - Chunk "specs": Thông số kỹ thuật (chỉ khi có specs)
    - Chunk "description": Mô tả chi tiết (chỉ khi không có specs VÀ mô tả > 200 chars)
    """
    chunks = []
    
    brand = prod.get('brand', prod.get('company', ''))
    series = prod.get('series', '')
    device = prod.get('device', '')
    category = prod.get('category', '')
    price = prod.get('price_formatted', '')
    company = prod.get('company', '')
    
    # Chunk 1: Identity — luôn có
    identity_parts = [f"Thương hiệu: {brand}", f"Loại thiết bị: {category}"]
    if series:
        identity_parts.append(f"Dòng sản phẩm: {series}")
    identity_parts.extend([
        f"Thiết bị: {device}",
        f"Nhà phân phối: {company}",
        f"Giá bán: {price}"
    ])
    chunks.append({"text": " | ".join(identity_parts), "type": "identity"})
    
    # Chunk 2: Specs — chỉ khi có structured specs
    specs = prod.get('specs', {})
    if specs:
        specs_parts = [f"{k}: {v}" for k, v in specs.items()]
        specs_text = f"{device} | Thông số kỹ thuật: " + " | ".join(specs_parts)
        chunks.append({"text": specs_text[:800], "type": "specs"})
    
    # Chunk 3: Description — chỉ khi KHÔNG có specs VÀ mô tả đủ dài
    description = str(prod.get('description', ''))
    if not specs and len(description) > 200:
        desc_text = f"{device} | Mô tả: {description[:800]}"
        chunks.append({"text": desc_text, "type": "description"})
    
    return chunks


# ============================================================
# MAIN INGEST FUNCTION
# ============================================================

def ingest_data():
    
    # 1. Kiểm tra file dữ liệu
    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parent
    data_path = project_root / "data.xlsx"
    if not data_path.exists():
        print(f"Lỗi: Không tìm thấy file {data_path}")
        return
        
    print("Đang đọc file Excel...")
    df = pd.read_excel(data_path)
    
    # Làm sạch dữ liệu rỗng
    df["Công ty"] = df["Công ty"].fillna("Không rõ")
    df["Thiết bị"] = df["Thiết bị"].fillna("Thiết bị không tên")
    df["Mô tả sản phẩm"] = df["Mô tả sản phẩm"].fillna("Không có mô tả chi tiết")
    df["Giá"] = df["Giá"].fillna(0).astype(int)
    df["image"] = df["image"].fillna("")
    
    print(f"Đã đọc thành công {len(df)} dòng sản phẩm.")
    
    # 2. Tạo nội dung văn bản để Embedding (Contextual Retrieval)
    metadata_list = []
    
    brand_stats = {}
    category_stats = {}
    
    for idx, row in df.iterrows():
        device = str(row["Thiết bị"])
        company = str(row["Công ty"])
        description = str(row["Mô tả sản phẩm"])
        
        # Trích xuất brand, category, series, specs
        brand = extract_brand(device, company)
        category = extract_category(device, description)
        series = extract_series(device)
        specs = extract_structured_specs(description)
        
        # Thống kê
        brand_stats[brand] = brand_stats.get(brand, 0) + 1
        category_stats[category] = category_stats.get(category, 0) + 1
        
        # Định dạng giá tiền VNĐ
        gia_vnd = "{:,}".format(row["Giá"]) + " VNĐ" if row["Giá"] > 0 else "Liên hệ"
        

        
        # Metadata mở rộng — thêm series và specs
        metadata_list.append({
            "id": idx,
            "company": company,
            "brand": brand,
            "category": category,
            "series": series,
            "device": device,
            "description": description,
            "specs": specs,
            "price": int(row["Giá"]),
            "price_formatted": gia_vnd,
            "image": row["image"]
        })
        
    # 3. In thống kê brand & category & series
    print(f"\n=== THỐNG KÊ BRAND (Top 20) ===")
    for b, c in sorted(brand_stats.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {b}: {c} sản phẩm")
    
    print(f"\n=== THỐNG KÊ CATEGORY ===")
    for cat, c in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {c} sản phẩm")
    
    # Thống kê series
    series_stats = {}
    specs_count = 0
    for m in metadata_list:
        s = m.get("series", "")
        if s:
            series_stats[s] = series_stats.get(s, 0) + 1
        if m.get("specs"):
            specs_count += 1
    print(f"\n=== THỐNG KÊ SERIES (Top 20) ===")
    for s, c in sorted(series_stats.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {s}: {c} sản phẩm")
    print(f"\nSố sản phẩm có structured specs: {specs_count}/{len(metadata_list)}")
    
    # 4. Tạo multi-chunks từ metadata
    print("\n=== TẠO MULTI-CHUNKS ===")
    all_chunks = []      # Text cho embedding (có prefix "passage: ")
    chunk_map = []       # Mapping chunk_idx → product_idx + chunk_type
    
    for prod_idx, prod in enumerate(metadata_list):
        product_chunks = create_product_chunks(prod)
        for chunk in product_chunks:
            chunk_map.append({
                "chunk_idx": len(all_chunks),
                "product_idx": prod_idx,
                "chunk_type": chunk["type"]
            })
            # Thêm prefix "passage: " — bắt buộc cho E5 model
            all_chunks.append(f"passage: {chunk['text']}")
    
    chunk_type_stats = {}
    for cm in chunk_map:
        t = cm["chunk_type"]
        chunk_type_stats[t] = chunk_type_stats.get(t, 0) + 1
    
    print(f"Tổng chunks: {len(all_chunks)} (từ {len(metadata_list)} sản phẩm)")
    for t, c in chunk_type_stats.items():
        print(f"  {t}: {c} chunks")
    avg_chunks = len(all_chunks) / len(metadata_list) if metadata_list else 0
    print(f"Trung bình: {avg_chunks:.1f} chunks/sản phẩm")
    
    # 5. Tải mô hình Embedding (E5 multilingual — tối ưu cho retrieval)
    print("\nĐang tải mô hình Embedding (intfloat/multilingual-e5-small)...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    
    # 6. Tính toán Vector Embeddings cho tất cả chunks
    print("Đang chuyển đổi chunks thành Vectors. Quá trình này có thể mất ít phút...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    
    # Chuẩn hóa các vector (để tính cosine similarity đơn giản bằng tích vô hướng)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normalized_embeddings = embeddings / norms
    
    # 7. Lưu trữ kết quả (Qdrant & Metadata)
    vector_store_dir = backend_dir / "vector_store"
    vector_store_dir.mkdir(exist_ok=True)
    
    metadata_json_path = vector_store_dir / "metadata.json"
    chunk_map_path = vector_store_dir / "chunk_map.json"
    
    print("\nĐang khởi tạo kết nối Qdrant Docker (localhost:6333)...")
    try:
        client = qdrant_client.QdrantClient(url="http://localhost:6333")
        
        print("Đang tạo collection 'products' trên Qdrant...")
        client.recreate_collection(
            collection_name="products",
            vectors_config=VectorParams(size=normalized_embeddings.shape[1], distance=Distance.COSINE)
        )
        
        print("Đang đẩy dữ liệu lên Qdrant...")
        points = []
        for i, (chunk_vector, chunk_info) in enumerate(zip(normalized_embeddings, chunk_map)):
            points.append(PointStruct(
                id=i,
                vector=chunk_vector.tolist(),
                payload=chunk_info
            ))
            # Push theo batch 1000 items
            if len(points) >= 1000:
                client.upload_points(collection_name="products", points=points)
                points = []
                print(f"  Đã đẩy {i+1} chunks...")
        if points:
            client.upload_points(collection_name="products", points=points)
            print(f"  Đã đẩy xong tất cả {len(all_chunks)} chunks.")
    except Exception as e:
        print(f"\nLỖI: Không thể kết nối hoặc đẩy dữ liệu lên Qdrant Docker. Lỗi: {e}")
        print("Vui lòng đảm bảo Qdrant Docker đang chạy (port 6333)!")
        return

    print(f"\nĐang lưu metadata JSON vào {metadata_json_path}...")
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False)
    
    print(f"Đang lưu chunk map vào {chunk_map_path}...")
    with open(chunk_map_path, "w", encoding="utf-8") as f:
        json.dump(chunk_map, f, ensure_ascii=False)
        
    print("\n--- HOÀN THÀNH INGEST DỮ LIỆU THÀNH CÔNG! ---")
    print(f"Tổng sản phẩm: {len(metadata_list)}")
    print(f"Tổng chunks: {len(all_chunks)}")
    print(f"Số brands phát hiện: {len(brand_stats)}")
    print(f"Số categories: {len(category_stats)}")

if __name__ == "__main__":
    ingest_data()
