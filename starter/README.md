# Starter · PostgreSQL + pgvector

Bộ khởi động **tùy chọn** cho lab cá nhân. Docker Compose tạo PostgreSQL + pgvector local; các file trong `src/` ban đầu là khung pseudocode có TODO.

> **Trạng thái trong bài nộp này:** `src/` đã được hiện thực đầy đủ, nhưng **không dùng** nhánh PostgreSQL + pgvector. Máy làm bài không có Docker, nên vector store là **Qdrant embedded** (`qdrant-client`, lưu trên đĩa tại `qdrant_data/`), embedding là model local `BAAI/bge-small-en-v1.5` qua fastembed. Vì vậy `docker-compose.yml`, `init.sql`, `.env.example` và `scripts/check_setup.py` vẫn còn trong repo như tài liệu tham khảo của starter, không nằm trong đường chạy. Lý do đổi và cách chạy thật: [report/individual/2A202604117_NguyenVanMinh.md](../report/individual/2A202604117_NguyenVanMinh.md) mục 1.3.

## Cấu trúc

```text
starter/
├── docker-compose.yml     # PostgreSQL 16 + pgvector, chỉ mở cổng local
├── init.sql               # bật extension khi tạo database mới
├── requirements.txt       # pymupdf, qdrant-client, fastembed, python-dotenv, numpy
├── .env.example           # cấu hình mẫu của starter (không dùng trong bài này)
├── scripts/check_setup.py # kiểm tra kết nối pgvector (không dùng trong bài này)
├── scripts/               # fetch_corpus, validate_benchmark, audit, chẩn đoán
├── src/
│   ├── models.py          # DocumentPage, Chunk, Block
│   ├── config.py          # đường dẫn, model, ba cấu hình A/B/C
│   ├── ingestion.py       # đọc pages.jsonl, layout block, heading
│   ├── chunking.py        # fixed window và paragraph/section packing
│   ├── embeddings.py      # fastembed (local ONNX), không fallback mock
│   ├── store.py           # Qdrant embedded: upsert idempotent, cosine + filter
│   ├── benchmark.py       # nạp và kiểm chứng bộ câu hỏi
│   ├── evaluation.py      # Hit@1 / Evidence Hit@3, trace từng câu
│   └── run.py             # TODO: kết nối các bước thành pipeline
├── data/                  # bạn tự chuẩn bị corpus được phép sử dụng
└── results/               # lưu kết quả của bạn
```

## 1. Cài và kiểm tra database

Cần Docker Desktop hoặc Docker Engine + Compose, Python **3.11–3.14**, và mạng để tải image/dependencies lần đầu. Các lệnh sau chạy từ thư mục `starter/`:

```bash
cd starter
python3 -m venv .venv
```

Kích hoạt `.venv`: macOS/Linux dùng `source .venv/bin/activate`; Windows PowerShell dùng `.venv\Scripts\Activate.ps1`. Sau đó:

```bash
python -m pip install -r requirements.txt
# macOS/Linux: cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
```

Kỳ vọng: hiện tên database, phiên bản PostgreSQL và pgvector, sau đó `Setup OK`. Đây **chỉ là kiểm tra hạ tầng**, không phải kết quả benchmark. Docker dùng cổng `127.0.0.1:55433` và mật khẩu cố định **chỉ dùng local cho lab**; nếu cổng bận, sửa đồng thời port trong `docker-compose.yml` và `DATABASE_URL` trong `.env`. `init.sql` chỉ chạy khi volume mới được tạo; nếu database sẵn có chưa bật extension, xem log Docker và kiểm tra cấu hình thay vì xóa volume chứa dữ liệu.

Dừng container nhưng giữ dữ liệu:

```bash
docker compose -p day6_student_starter stop
```

## 2. Chọn embedding model

Điền key tương ứng trong `.env` nếu dùng OpenRouter, OpenAI hoặc Gemini; **không commit `.env`**. Model local không cần API key. `requirements.txt` chỉ có dependencies cơ bản: tự cài SDK hoặc thư viện local phù hợp với model bạn chọn và ghi lại version/lệnh cài trong báo cáo. Ghi tên model và số chiều; dùng cùng model cho toàn bộ chiến lược và vector query tương thích vector document.

## 3. Pipeline trong bài nộp

Corpus nằm ở `../data/` (manifest + text đã chuẩn hóa), không nằm trong `starter/data/`.

```bash
# từ thư mục gốc repo
python starter/scripts/fetch_corpus.py        # tải PDF, xác minh CC BY 4.0, sinh manifest + pages.jsonl
python starter/scripts/validate_benchmark.py  # evidence phải là substring thật trên trang gold
cd starter && python -m src.run --recreate    # chunk -> embed -> Qdrant -> benchmark
```

`python -m src.run --skip-ingest` chạy lại benchmark trên vector đã lưu. Không cần API key.
Chi tiết tham số, kết quả và phần audit: xem báo cáo ở thư mục `report/individual/`.
