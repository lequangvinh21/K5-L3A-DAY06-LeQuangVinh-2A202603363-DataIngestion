# Starter · PostgreSQL + pgvector

Bộ khởi động **tùy chọn** cho lab cá nhân. Docker Compose tạo PostgreSQL + pgvector local; pipeline trong `src/` đọc PDF, tạo embedding local, lưu/tìm vector và chạy benchmark. Có thể thay PostgreSQL bằng Qdrant, Chroma hoặc vector store tương đương theo [GUIDE.md](../GUIDE.md).

## Cấu trúc

```text
starter/
├── docker-compose.yml     # PostgreSQL 16 + pgvector, chỉ mở cổng local
├── init.sql               # bật extension khi tạo database mới
├── requirements.txt       # psycopg, pypdf, python-dotenv và sentence-transformers
├── .env.example           # cấu hình mẫu; copy thành .env
├── scripts/check_setup.py # kiểm tra kết nối và extension (đã chạy được)
├── src/
│   ├── models.py          # cấu trúc dữ liệu mẫu
│   ├── ingestion.py       # đọc / chuẩn hóa PDF và giữ metadata trang
│   ├── chunking.py        # fixed-size và paragraph-aware
│   ├── embeddings.py      # model Sentence Transformers chạy local
│   ├── store.py           # pgvector, metadata filter và cosine search
│   ├── evaluation.py      # retrieval metrics và trace
│   └── run.py             # orchestration, báo cáo và mẫu chấm thủ công
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

Pipeline mặc định dùng model local `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều), nên không cần API key và nội dung tài liệu không được gửi tới dịch vụ embedding bên ngoài. Lần chạy đầu cần internet để tải model; những lần sau dùng bản cache local. Có thể đặt `EMBEDDING_MODEL` trong `.env` để chọn model Sentence Transformers khác, nhưng cần ghi model, phiên bản thư viện, số chiều và cách xử lý query/document trong báo cáo. Giữ nguyên model cho cả ba chiến lược. **Không commit `.env`**.

## 3. Hoàn thiện pipeline

1. Đặt corpus và manifest ở `data/`. Đọc [GUIDE.md](../GUIDE.md) về **30 trang nội dung**, quyền dữ liệu và metadata.
2. Chọn model local mặc định hoặc đặt `EMBEDDING_MODEL` trong `.env`; ghi model và dependency version vào báo cáo.
3. Chạy `python -m src.run --split dev` từ thư mục `starter/`. Lần đầu cần internet để tải model; PDF/manifest và benchmark hiện tại được đọc từ thư mục gốc `data/`.
4. Chọn chiến lược trên dev. Sau khi cố định cấu hình, truyền chiến lược đã chọn khi chạy test, ví dụ `python -m src.run --split test --selected-strategy A`; pipeline lưu trace và `manual_review_template.json` trong thư mục kết quả.
5. Hoàn tất nhãn chấm thủ công và báo cáo đủ năm tiêu chí theo [BENCHMARK.md](../BENCHMARK.md).

Pipeline tạo ba cấu hình: A fixed-size (500 ký tự, overlap 50), B theo đoạn (tối đa 500 ký tự), C giống B nhưng chỉ áp dụng metadata filter khi câu hỏi có filter. Tìm kiếm dùng cosine exact. Kết quả được lưu trong thư mục con có timestamp của `results/`. Lưu code, cấu hình, kết quả và lệnh chạy để người chấm tái lập; không commit key, dữ liệu mật hoặc dữ liệu không có quyền chia sẻ.
