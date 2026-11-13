# Báo cáo cá nhân — Day 6 Data Ingestion

- **Họ tên:** Lê Quang Vinh
- **MSSV:** 2A202603363
- **Lớp:** L3A
**Chủ đề:** Business Law — *Business Law I Essentials*

## Dữ liệu và pipeline

Corpus là PDF *Business Law I Essentials* của OpenStax/Rice University, lấy từ [trang sách OpenStax](https://openstax.org/details/books/business-law-i-essentials), giấy phép **CC BY-NC-SA 4.0**. Khi phân phối lại phải ghi công, dùng phi thương mại và tuân thủ điều khoản ShareAlike. [Manifest](../../data/manifest.json) ghi 165 trang vật lý, trong đó 131 trang nội dung hữu ích; loại trang bìa/mục lục, trang tham khảo, bibliography, đáp án và index theo quy tắc đã mô tả trong manifest. Số trang trong kết quả là số trang vật lý của PDF.

PDF được chuẩn hóa Unicode NFC và khoảng trắng, giữ đoạn văn và liên kết trang. Không dùng OCR. Có một cảnh báo không trích được text ở trang vật lý 2; trang này thuộc front matter, không nằm trong 131 trang hữu ích. Xem [log cảnh báo dev](../../results/dev/extraction_warnings.json) và [log cảnh báo test](../../results/test/extraction_warnings.json).

Embedding chạy local bằng Sentence Transformers `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, thư viện `sentence-transformers` 5.7.0, vector 384 chiều. Query và document dùng cùng phương thức encode và L2 normalization. Vector store là PostgreSQL 16.16 + pgvector 0.8.7, bảng `day6_chunks_384`, exact cosine search; chi phí dịch vụ embedding là 0. Python 3.12.10, pypdf 6.21.4, psycopg 3.3.8. Lệnh chạy từ `starter/`:

```powershell
python -m src.run --split dev
python -m src.run --split test --selected-strategy A
```

Metadata chunk gồm `document_id`, `source`, `page`, `page_id`, `chapter`, chiến lược và offset. `chapter` là số nguyên suy ra từ trang bắt đầu chương trong PDF, dùng làm filter khi câu hỏi cung cấp chương. Sau lần chạy test, số bản ghi trong bảng là A=764, B=712, C=712 — đúng với số chunk trong run metadata; mã nạp lại xóa chunk cùng tài liệu/chiến lược trước khi nạp, tránh tích lũy bản sao.

## Cấu hình và benchmark

| Cấu hình | Chunking | Filter |
|---|---|---|
| A | Cửa sổ cố định 500 ký tự, overlap 50 | Không |
| B | Theo đoạn văn, tối đa 500 ký tự | Không |
| C | Giống B | Lọc `chapter` khi câu hỏi có cung cấp chương |

B–C chỉ khác filter; A–B khác cách chunking. Cả ba dùng cùng corpus, embedding, store, metric cosine và benchmark. Giả thuyết, cấu hình chi tiết và chỉ số được ghi trong [experiment log](../../artifacts/experiment_log.csv), còn tham số chạy thực tế trong [dev metadata](../../results/dev/run_metadata.json) và [test metadata](../../results/test/run_metadata.json).

[Benchmark](../../data/benchmark.json) có 20 câu đã chia 15 dev / 5 test. Có ba câu filter ở dev (q02, q05, q10) và hai câu filter ở test (q16, q18). Evidence và trang gold được kiểm chứng với PDF. Tiêu chí chọn đã cố định trước khi dùng test: ưu tiên Evidence Hit@3 cao nhất trên dev, rồi Hit@1, sau đó tài nguyên thấp hơn. Vì A đạt Evidence Hit@3 10/15, cao hơn B/C 9/15, chọn A trước khi chạy test; không điều chỉnh cấu hình theo kết quả test.

## Kết quả

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Số chunk | Chunk + embed + store (s) | Chi phí |
|---|---|---:|---:|---:|---:|---:|
| A | Dev | 6/15 (0.400) | 10/15 (0.667) | 764 | 46.1187 | 0 |
| B | Dev | 7/15 (0.467) | 9/15 (0.600) | 712 | 41.9503 | 0 |
| C | Dev | 7/15 (0.467) | 9/15 (0.600) | 712 | 41.1655 | 0 |
| A | Test | 3/5 (0.600) | 4/5 (0.800) | 764 | 45.5546 | 0 |
| B | Test | 3/5 (0.600) | 3/5 (0.600) | 712 | 42.9141 | 0 |
| C | Test | 3/5 (0.600) | 3/5 (0.600) | 712 | 41.2394 | 0 |

Thời gian trong bảng là tổng chunking, embedding và storage theo từng run; ingestion PDF riêng mất 2.6655 giây ở dev và 2.6372 giây ở test. Đây là một lần đo mỗi split, phụ thuộc cache/tải máy, không phải benchmark hiệu năng ổn định. Chi tiết metric, phiên bản và tài nguyên nằm trong [dev run metadata](../../results/dev/run_metadata.json) và [test run metadata](../../results/test/run_metadata.json).

### Nhãn audit

| Strategy | Grounding 0 / 1 / 2 | Trung bình /2 | Metadata đúng, đủ | Filter tuân thủ | Coherence trung bình /2 |
|---|---:|---:|---:|---:|---:|
| A | 0 / 1 / 4 | 1.8 | 15/15 | 2/2 câu hỏi (6/6 kết quả top-3) | 1.0 (10 mẫu) |
| B | 0 / 1 / 4 | 1.8 | 15/15 | 2/2 câu hỏi (6/6 kết quả top-3) | 1.0 (10 mẫu) |
| C | 0 / 1 / 4 | 1.8 | 15/15 | 2/2 câu hỏi (6/6 kết quả top-3) | 1.0 (10 mẫu) |

Metadata được đối chiếu với trace và manifest: source, trang, page ID, chapter, strategy và offset. Trường `chapter` giúp giới hạn kết quả theo chương; hạn chế là giá trị được suy ra từ ranh giới chương theo trang, nên phụ thuộc cấu trúc của PDF này. Mỗi cấu hình có 15 bản ghi audit (5 câu × top-3); hai câu có filter, q16 và q18, đều có cả top-3 thuộc đúng chương.

Coherence dùng cùng 10 vị trí trên các cấu hình: trang PDF 13–21 và 23, tại các offset lần lượt 582, 1003, 1027, 2056, 1052, 2018, 1576, 892, 1042 và 888. Cả 30 chunk đều được chấm 1: nội dung nhìn chung đọc được nhưng có điểm bị cắt giữa câu hoặc thiếu ngữ cảnh ở ranh giới. Bảng nhãn và lý do từng mẫu nằm trong [manual review template](../../results/test/manual_review_template.json).

**Phương pháp audit:** các nhãn đã được điền đầy đủ bằng đánh giá có AI hỗ trợ (Copilot), bằng cách đọc nội dung top-3 và so với câu hỏi/đáp án mong đợi; không dùng similarity score để cho điểm. Đây không phải đánh giá độc lập bởi người. Khi nộp, cần khai báo việc AI hỗ trợ và sửa nhãn nào người học không đồng ý.

## Phân tích và kết luận

- **A–B:** B có Hit@1 cao hơn dev 1/15, nhưng Evidence Hit@3 thấp hơn 1/15; trên test Hit@1 bằng A, còn Evidence Hit@3 thấp hơn 1/5. A tạo thêm 52 chunk và 32,042 ký tự chunk so với B; coherence được chấm bằng nhau. Chọn A theo tiêu chí Evidence Hit@3 đã đặt trước.
- **B–C:** B và C có cùng số chunk và cùng Hit@1/Evidence Hit@3 ở cả dev lẫn test. Với tập câu hỏi này, filter không làm thay đổi top-3; việc top-3 vẫn đúng chương không chứng minh filter cải thiện độ chính xác.
- **Ví dụ lỗi q18:** các cấu hình không truy xuất được đoạn cụ thể trên trang 151 nói mục tiêu minh bạch là cập nhật công chúng về thay đổi có thể ảnh hưởng giá chứng khoán. Một số kết quả, như trang 147, chỉ nêu mục tiêu rộng hơn là giúp nhà đầu tư ra quyết định; vì thiếu chi tiết cần thiết, grounding được chấm 1.
- **Ví dụ lỗi q20 ở B/C:** bằng chứng về “hơn hai bên” và phần “phức tạp, tốn thời gian, khó giải quyết” nằm rải giữa các chunk top-3 ở trang 24. Kết hợp các chunk vẫn đủ trả lời nên audit grounding là 2, nhưng không chunk riêng lẻ nào chứa trọn evidence quote; đây là giới hạn của Evidence Hit@3 tự động theo quy tắc một chunk.

Test chỉ có 5 câu trên một corpus nên kết quả minh họa, không chứng minh A tốt nhất nói chung. Coherence mẫu nhỏ; nhãn grounding có AI hỗ trợ; thời gian chỉ có một lần đo mỗi split. A thắng theo mục tiêu evidence retrieval nhưng dùng nhiều chunk hơn. Nếu làm tiếp, nên cải thiện ranh giới chunk để tránh cắt câu và bổ sung câu hỏi test độc lập, đồng thời giữ riêng đánh giá bằng chứng nhiều chunk với định nghĩa Evidence Hit@3 hiện tại.

## Trace và artifacts

- [Trace dev — 15 câu × 3 cấu hình](../../results/dev/traces_dev.json)
- [Trace test — 5 câu × 3 cấu hình](../../results/test/traces_test.json)
- [Nhãn audit test](../../results/test/manual_review_template.json)
- [Experiment log A/B/C](../../artifacts/experiment_log.csv)
- [Manifest](../../data/manifest.json), [benchmark và gold](../../data/benchmark.json)
- Mã pipeline: [starter/src/](../../starter/src/); hướng dẫn chạy: [starter/README.md](../../starter/README.md)
