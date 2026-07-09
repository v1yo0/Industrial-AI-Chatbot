# Industrial AI Chatbot (RAG)

Chatbot tư vấn thiết bị công nghiệp cho dữ liệu sản phẩm trong `data.xlsx`. Hệ thống dùng FastAPI, SentenceTransformers và vector database **Qdrant (chạy qua Docker)** để truy xuất sản phẩm liên quan, sau đó dùng Gemini API để tổng hợp câu trả lời tiếng Việt.

## Tính năng

- Tra cứu thông số kỹ thuật, giá và hình ảnh sản phẩm từ dữ liệu Excel.
- Tìm kiếm hybrid: lọc theo hãng/model/loại thiết bị kết hợp cosine similarity và BM25.
- Giao diện Floating Chat Widget (nút chat nổi ở góc) nhỏ gọn, HTML/CSS/JS thuần, có hiển thị card sản phẩm dạng carousel vuốt ngang và modal popup xem chi tiết.
- Trả lời bằng tiếng Việt, tích hợp truyền tải dữ liệu theo thời gian thực (Real-time Token-by-Token Streaming).
- Hỗ trợ nhiều Gemini API keys qua biến `GEMINI_API_KEYS`.

## Lưu ý bảo mật

- Không commit file `backend/.env` thật.
- Tạo file `backend/.env` từ `backend/.env.example` và điền API key của bạn.
- Nếu API key từng bị chia sẻ hoặc commit, hãy revoke/rotate key trên Google AI Studio.

## Cài đặt

### 1. Cài thư viện backend

```bash
python -m pip install -r backend/requirements.txt
```

### 2. Cấu hình Gemini

Sao chép file mẫu:

```bash
copy backend\.env.example backend\.env
```

Sau đó sửa `backend/.env`:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=gemini-2.5-flash
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

Có thể dùng nhiều key:

```env
GEMINI_API_KEYS=KEY_1,KEY_2,KEY_3
```

### 3. Cài đặt và khởi chạy Qdrant (Vector Database)

Hệ thống sử dụng Qdrant lưu trữ dưới dạng Docker container.

```bash
docker run -d -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

*Qdrant Web UI (Dashboard) sẽ được host ở `http://localhost:6333/dashboard`.*

### 4. Tạo vector store
Download dataset : "https://drive.google.com/drive/folders/1iiTd0LejstHElslXODa-8Y0S0oMBul7M?usp=sharing"

Script sẽ đọc `data.xlsx` ở thư mục gốc project và ghi dữ liệu vector lên database Qdrant.

```bash
python backend/ingest.py
```

Lần đầu chạy có thể mất vài phút vì cần tải model embedding `intfloat/multilingual-e5-small`.

### 5. Chạy backend

```bash
python backend/app.py
```

Backend mặc định chạy tại `http://127.0.0.1:8000`.

### 6. Chạy frontend

Khuyến nghị chạy static server:

```bash
python -m http.server 3000 --directory frontend
```

Mở trình duyệt tại `http://localhost:3000`.

## Cấu trúc

- `backend/app.py`: FastAPI app, retrieval (Qdrant), query parsing và Gemini client.
- `backend/ingest.py`: đọc Excel, tạo embeddings và đẩy lên Qdrant.
- `backend/.env.example`: mẫu cấu hình môi trường.
- `frontend/index.html`: giao diện chat.
- `frontend/styles.css`: style giao diện.
- `frontend/app.js`: gọi API, render chat/card/modal.
- `data.xlsx`: dữ liệu sản phẩm gốc.

## Hình ảnh giao diện

![Giao diện 1](image/1.png)

![Giao diện 2](image/2.png)

## Video Demo

![Demo](demo.gif)

*(Lưu ý: Để video có thể tự động phát ngay trong Markdown giống như file mẫu, bạn cần chuyển đổi file `demo.mp4` thành file `demo.gif` rồi đặt ở thư mục gốc nhé. Nếu dùng file mp4 thì bạn chỉ có thể để dạng link tải về như sau: [Tải Video Demo](demo.mp4))*
