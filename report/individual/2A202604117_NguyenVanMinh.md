# Báo cáo cá nhân — Day 6 · Data Ingestion Engineering

Nguyễn Văn Minh · 2A202604117 · L3B

Số liệu trong báo cáo lấy từ các file trong `results/` do pipeline sinh ra. Lệnh tái lập ở mục 1.5.

## 1. Dữ liệu và pipeline

### 1.1 Chủ đề và nguồn

Corpus gồm 14 preprint arXiv về reinforcement learning và điều khiển tối ưu. Mỗi tài liệu được
kiểm tra giấy phép trên trang abstract trước khi tải; `starter/scripts/fetch_corpus.py` dừng nếu
một tài liệu không phải CC BY 4.0.

| Hạng mục | Giá trị |
|---|---|
| Số tài liệu | 14 (`data/manifest.json`) |
| Trang vật lý | 328 |
| Trang nội dung | 264 |
| Ký tự nội dung | 803.977 |
| Năm (theo mã arXiv) | 2020, 2021, 2022, 2023, 2024, 2025 |
| Nhóm | cs.LG (10), eess.SY (2), math.OC (1), stat.ML (1) |

Manifest ghi cho từng tài liệu: `source_id`, `title`, `source_path_or_url`, `pdf_url`,
`license_or_permission`, `accessed_at`, `primary_category`, `year`, `document_type`,
`language`, `physical_pages`, danh sách `content_pages`, `content_chars`, `sha256_pdf`,
`dropped_running_headers` và `page_rule`.

Quy tắc tính trang, ghi trong `page_rule` của từng tài liệu: trang nội dung là trang vật lý có ít
nhất 1.000 ký tự text đã chuẩn hóa, sau khi loại khối bibliography (từ heading References đến
heading appendix hoặc acknowledgement kế tiếp) và loại trang trắng hoặc chỉ có hình.

Hai phiên bản trước của quy tắc này bị bỏ vì không khớp với dữ liệu:

- Cắt mọi trang sau heading References làm mất toàn bộ appendix của `2206.06841v1` (12 trang còn
  4). Bản hiện tại dừng khối bibliography ở heading appendix kế tiếp.
- Phân loại trang bibliography theo tỉ lệ dòng khớp mẫu citation không dùng được.
  `starter/scripts/calibrate_bib_rule.py` cho thấy trang reference thật chỉ đạt tỉ lệ 0,02 đến
  0,50 vì mỗi entry bị wrap nhiều dòng, trong khi vài trang proof bị chấm nhầm.

`starter/scripts/audit_page_rules.py` in ra từng trang bị loại kèm dòng đầu của trang để kiểm tra
lại bằng mắt.

Giới hạn đã biết:

- PDF hai cột có công thức nên text bị xé theo layout block (ví dụ `xM` và `r` nằm khác dòng).
  Đây là giới hạn của bước extract, không phải của chunker.
- Quy tắc loại trang sai theo hướng thận trọng ở hai tài liệu: `2410.14069v1` p15 và p16 (NeurIPS
  Paper Checklist), `2506.17155v2` p17 và p18 (Supplementary Materials) bị tính là bibliography.
  Vì vậy 264 là cận dưới của số trang nội dung.
- Không có trang scan nên không cần OCR.

### 1.2 Chuẩn hóa

`normalize_text` dùng chung ở bước extract và ở pipeline: NFKC, bỏ soft hyphen, nối từ bị gạch nối
cuối dòng (`con-\nstraint` thành `constraint`), gộp khoảng trắng trong dòng, giữ dòng trống làm
ranh giới đoạn. `strip_running_headers` bỏ dòng lặp trên ít nhất 50% số trang, tức header, footer
và số trang.

Hệ quả cần biết khi đọc evidence: `real-\nworld` thành `realworld`, nên câu hỏi `q03` dùng đúng
dạng `realworld`. Evidence được kiểm chứng bằng máy nên lỗi này bị bắt ở bước validate.

Bước extract dùng layout block (`page.get_text("blocks")`) chứ không dùng text phẳng. Text phẳng
của bài hai cột không có dòng trống giữa các đoạn, nên cả trang thành một đoạn và toàn bộ heading
bị mất: lần chạy đầu chỉ phát hiện 11 section trên 264 trang. Sau khi chuyển sang block, pipeline
phát hiện 229 section ở cấu hình A và 255 ở B và C.

### 1.3 Embedding, vector store, metadata

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| Embedding | `BAAI/bge-small-en-v1.5`, 384 chiều, fastembed 0.9.1 (ONNX, local) | document dùng `embed`, query dùng `query_embed` để áp prefix retrieval của model |
| Vector store | `qdrant-client` 1.20.0, embedded persistent tại `qdrant_data/`, cosine | local mode tìm kiếm chính xác, không approximate |
| Top-k | 3 | |
| Chi phí | 0 USD | model chạy local, không gọi API |
| Thời gian | ingest ba cấu hình 651 s; toàn bộ run 1.326 s, gồm lần nạp lại để kiểm tra idempotency | `results/run_meta.json` |

Lựa chọn đầu tiên là API Mistral (`mistral-embed`, 1.024 chiều). Key có sẵn trả về
`HTTP 401: Your API key expired on 2026-11-14`, nên `starter/src/embeddings.py` dừng và báo lỗi
thay vì fallback, sau đó chuyển sang model local. Không có vector mock ở bất kỳ bước nào.

Metadata mỗi chunk: `document_id`, `source`, `page`, `section`, `year`, `primary_category`,
`document_type`, `language`, `chunk_index`, `char_len`. Hai trường dùng để lọc là `year` (6 giá trị,
lấy từ mã arXiv, không suy đoán ngày hiệu lực) và `primary_category` (4 giá trị, lấy từ metadata
arXiv). `section` dùng để truy vết, không dùng để lọc, vì heading do regex phát hiện nên có thể sai.

Point ID là UUID5 của `chunk_id`, mà `chunk_id` là hash ổn định của strategy, document, page, index
và text. Chunk cũ không còn tồn tại bị xoá theo cặp `(strategy, document_id)`. Kiểm chứng: nạp lại
cả ba cấu hình, số point giữ nguyên 3.205.

### 1.4 Ba cấu hình

| | Chunk | Metadata và truy xuất |
|---|---|---|
| A | cửa sổ cố định 800 ký tự, overlap 150, trong từng trang | đủ metadata, không lọc |
| B | gom đoạn theo layout block và heading, tối đa 1.200 ký tự, không cắt qua heading, trong từng trang | đủ metadata, không lọc |
| C | giống hệt B | thêm lọc `year` hoặc `primary_category` khi câu hỏi cung cấp |

A và B chỉ khác cách chunk; B và C chỉ khác lọc. C dùng lại đúng chunk của B nhưng được nạp riêng
để trace của C độc lập, 949 chunk mỗi cấu hình.

Chunk không vượt ranh giới trang. Mục đích là giữ `page` trong metadata chính xác tuyệt đối khi
chấm hit. Đánh đổi là câu vắt qua hai trang bị cắt; hệ quả định lượng ở mục 4.3.

### 1.5 Lệnh chạy

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r starter/requirements.txt
python starter/scripts/fetch_corpus.py            # tải PDF, xác minh license, sinh manifest + pages.jsonl
cd starter && ../.venv/Scripts/python.exe -m src.run --recreate
../.venv/Scripts/python.exe scripts/retrieval_depth.py
../.venv/Scripts/python.exe scripts/build_audit_samples.py
../.venv/Scripts/python.exe scripts/apply_audit_labels.py
```

`data/corpus/*.pdf` không commit (20 MB, tải lại bằng script đầu tiên). `data/pages.jsonl` (2,1 MB)
được commit để chấm lại evidence mà không cần tải PDF.

## 2. Cấu hình và benchmark

Bộ câu hỏi có 20 câu trong `benchmark/questions.jsonl`, chia 15 dev và 5 test trước khi chạy. Loại
câu: 11 trực tiếp, 4 diễn đạt lại, 2 điều kiện hoặc ngoại lệ, 3 câu metadata filter (2 dev, 1 test).
Mỗi câu có `id`, `split`, `type`, `question`, `expected_answer`, `gold_source_id`, `gold_pages`,
`evidence_quote`, `filter`, `notes`.

`starter/src/benchmark.py` kiểm chứng bộ câu hỏi trước khi truy xuất. Năm điều kiện: đủ trường bắt
buộc; trang gold là trang nội dung; `evidence_quote` là substring thật trên trang gold sau khi
chuẩn hóa khoảng trắng; giá trị filter tồn tại trong corpus; và đủ số câu theo từng split, đủ câu
filter. Kết quả trong `results/benchmark_validation.json` là `ok: true`, 0 lỗi. Nếu evidence không
khớp, pipeline dừng trước khi chạy truy xuất.

Filter là input do câu hỏi cung cấp trước khi truy xuất (ví dụ "In the 2024 paper…" ứng với
`year=2024`), không suy ra từ gold source.

Tiêu chí chọn cấu hình: ưu tiên (1) Evidence Hit@3 trên dev, (2) grounding thủ công, (3) chi phí và
tài nguyên. Tiêu chí này lấy từ GUIDE và được đặt cùng lúc với thiết kế ba cấu hình. Tôi không gọi
đây là pre-registration: kết quả dev và test được xem trong cùng một lần chạy, nên tập test không
phải holdout độc lập. Nó là mẫu kiểm tra 5 câu và không được dùng để chỉnh tham số nào.

## 3. Kết quả

### 3.1 Truy xuất

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Gold page trong top-3 | Số chunk |
|---|---|---|---|---|---|
| A | dev | 5/15 (33,3 %) | 8/15 (53,3 %) | 10/15 | 1.307 |
| A | test | 2/5 (40,0 %) | 4/5 (80,0 %) | 5/5 | |
| A | all | 7/20 (35,0 %) | 12/20 (60,0 %) | 15/20 | |
| B | dev | 7/15 (46,7 %) | 10/15 (66,7 %) | 11/15 | 949 |
| B | test | 2/5 (40,0 %) | 3/5 (60,0 %) | 4/5 | |
| B | all | 9/20 (45,0 %) | 13/20 (65,0 %) | 15/20 | |
| C | dev | 8/15 (53,3 %) | 10/15 (66,7 %) | 11/15 | 949 |
| C | test | 2/5 (40,0 %) | 3/5 (60,0 %) | 4/5 | |
| C | all | 10/20 (50,0 %) | 13/20 (65,0 %) | 15/20 | |

Nguồn: `results/metrics.json`; trace đầy đủ trong `results/traces_{A,B,C}.jsonl`.

### 3.2 Bốn khía cạnh còn lại

| Strategy | Coherence trung bình /2 (n=10) | Metadata đủ và đúng (n/N) | Filter tuân thủ | Grounding (n=5) và trung bình |
|---|---|---|---|---|
| A | 1,0 (10 nhãn 1) | 15/15 | 2/3 (dev 1/2, test 1/1) | 0 nhãn 0, 1 nhãn 1, 4 nhãn 2; trung bình 1,8 |
| B | 1,5 (5 nhãn 2, 5 nhãn 1) | 15/15 | 2/3 (dev 1/2, test 1/1) | 0 nhãn 0, 1 nhãn 1, 4 nhãn 2; trung bình 1,8 |
| C | 1,5 (giống B) | 15/15 | 3/3 (dev 2/2, test 1/1) | 0 nhãn 0, 1 nhãn 1, 4 nhãn 2; trung bình 1,8 |

Coherence chấm trên 10 vị trí cố định trước khi chấm điểm: trang 1 của 10 tài liệu đầu theo
`source_id`. Mỗi vị trí chấm chunk chứa phần đầu trang đó. Nhãn và lý do từng nhãn nằm trong
`results/audit_coherence.csv`.

Metadata kiểm tra bằng máy, không bằng mắt. Với mỗi bản ghi top-3 của 5 câu test (15 bản ghi mỗi
cấu hình), script xác nhận `source`, `year`, `primary_category` khớp manifest và 200 ký tự đầu của
chunk thật sự nằm trên trang mà metadata khai báo. Kết quả 45/45 bản ghi đúng
(`results/audit_metadata.csv`).

Grounding chấm thủ công top-3 của 5 câu test cho từng cấu hình (`results/audit_grounding.csv`).
Người chấm là tác giả; rủi ro thiên lệch nêu ở mục 4.5.

Về filter: A và B vi phạm 1 trong 2 câu dev (q14 lấy về chunk `year=2025`), C đúng 2/2. Ở test (q19)
cả ba cấu hình đều đúng, nhưng đúng vì top-3 đều thuộc tài liệu 2025 chứ không phải nhờ filter. C
tuân thủ 3/3 theo thiết kế, vì filter được áp ngay trong truy vấn; con số này chỉ có ý nghĩa khi so
với A và B.

### 3.3 Chẩn đoán độ sâu truy xuất

`results/retrieval_depth.json` tìm top-20 thay vì top-3:

| Strategy | Evidence ở top-1 | top-3 | top-10 | top-20 | Trang gold ở top-20 |
|---|---|---|---|---|---|
| A | 7 | 12 | 14 | 14 | 17 |
| B | 9 | 13 | 14 | 15 | 17 |
| C | 10 | 13 | 14 | 15 | 17 |

Với 17/20 câu, trang gold nằm trong top-20 nhưng chunk chứa đúng câu evidence thì không. Phần lớn
câu miss do xếp hạng trên corpus đồng nhất (14 bài cùng chủ đề, abstract nào cũng nói về offline
RL), không do chunking.

`starter/scripts/explain_misses.py` kiểm tra riêng từng câu miss trên trang gold:

| Câu | Kết luận |
|---|---|
| q01, q04, q07, q09, q10 | evidence nằm nguyên trong một chunk nhưng bị xếp dưới; q01 có cosine 0,687 so với 0,791 của top-1 |
| q06 (cấu hình B) | trang gold có 3 chunk nhưng không chunk nào chứa trọn câu evidence, tức lỗi chunking |

## 4. Phân tích và kết luận

### 4.1 A và B: đổi cách chunk

Trên dev, B hơn A ở cả hai chỉ số (Hit@1 5 lên 7, Hit@3 8 lên 10) và coherence 1,0 lên 1,5. Cửa sổ
800 ký tự của A bắt đầu ngay ở tiêu đề và tác giả, nên phần lớn chunk là tiêu đề, tác giả và nửa
đầu abstract: đọc hiểu được nhưng ý bị cụt, và phần tiêu đề làm loãng vector. B gom nguyên abstract
nên 5 trong 10 vị trí mẫu đạt 2 điểm.

B không hơn A ở mọi mặt. Test Hit@3 của B thấp hơn A (3/5 so với 4/5), chênh đúng một câu: q20 có
evidence ở rank 3 với A, tụt xuống rank 4 với B và C. Với n=5, đây là khác biệt một câu và không đủ
để kết luận A tốt hơn B; tôi không dùng nó để đổi lựa chọn.

Đọc theo độ sâu thì A sang B đổi chỗ đúng ba câu: q10 từ không có trong top-20 lên rank 1, q14 từ
không có trong top-20 lên rank 2, q20 tụt từ 3 xuống 4.

B còn sinh ra 5 chunk không chứa câu nào, chỉ có tiêu đề, tác giả, địa chỉ và chữ "Abstract", trên
14 tài liệu: `2104.10383v1` (154 ký tự), `2303.13117v1`, `2305.03360v1`, `2308.14897v1`,
`2505.22866v1`. Nguyên nhân là abstract dài hơn 1.200 ký tự nên không gộp được vào block tiêu đề
(`starter/scripts/count_title_only.py`). Cách sửa ở mục 4.6.

### 4.2 B và C: đổi lọc metadata

B và C dùng cùng chunk, nên mọi khác biệt đến từ filter. Hit@3 trên dev không đổi (10/15), Hit@1
trên dev tăng 7 lên 8, độ tuân thủ filter 2/3 lên 3/3. Tác động nhỏ vì chỉ 3 trong 20 câu có filter.
Kết quả này cho thấy filter không cải thiện khả năng tìm ra evidence, chỉ cải thiện thứ hạng và độ
đúng phạm vi: ở B, rank 1 của q14 là chunk `year=2025` (`2505.22866v1` p32) trong khi chunk đúng ở
rank 2; ở C chunk sai phạm vi bị loại nên rank 1 là đúng trang gold.

### 4.3 Hai ví dụ sai cụ thể

1. q06, cấu hình B, lỗi chunking. Câu hỏi về cách regret của thuật toán admission control scale
   theo số job tối đa. Trang gold `2308.02391v1` p1 có 3 chunk nhưng không chunk nào chứa trọn câu
   evidence, vì câu bị cắt qua ranh giới layout block. Top-3 trả về p1, p19 và p18 của cùng tài
   liệu. Ở A, q06 cũng miss nhưng vì lý do khác: evidence nguyên vẹn trong một chunk ở rank 7, chỉ
   ngoài top-3.
2. q01, cấu hình A, lỗi xếp hạng. Evidence nguyên vẹn trong một chunk của `2004.01625v2` p1, nhưng
   top-3 toàn bộ thuộc `2506.17155v2`. Cosine của chunk đúng là 0,687, của top-1 là 0,791. Tăng
   top-k không giải quyết được: evidence không có trong top-20.

### 4.4 Chiến lược chọn và đánh đổi

Chọn B làm mặc định, dùng C khi người dùng cung cấp được phạm vi năm hoặc nhóm. Căn cứ: tiêu chí
đã đặt là Hit@3 trên dev, B và C đạt 10/15 so với 8/15 của A; coherence của B là 1,5 so với 1,0;
và B dùng ít hơn 27% số chunk (949 so với 1.307) nên rẻ hơn khi ingest và khi mở rộng. C không đổi
Hit@3 so với B nhưng tăng độ đúng phạm vi, nên dùng C khi có filter.

Đánh đổi phải trả:

- Chunk theo trang nên câu vắt trang bị cắt, dẫn tới q06. Chunk xuyên trang sẽ sửa được nhưng phải
  đổi cách chấm hit từ một trang sang một khoảng trang.
- Chunk dài hơn (trung bình 838 so với 729 ký tự) nên tốn context hơn khi đưa vào LLM.
- Grounding không phân biệt được ba cấu hình, cả ba đều 1,8/2 với n=5. Trên mẫu này, khác biệt
  Hit@3 không chuyển thành khác biệt grounding: q20 ở B không khớp evidence nguyên văn nhưng top-1
  vẫn nêu đúng "sample complexity bounds under latent pushforward coverability (Theorem 3.2)", đủ
  để trả lời. Đây là lý do Evidence Hit@3 chỉ nên dùng như proxy, không thay nhãn thủ công.

### 4.5 Hạn chế

- Cỡ mẫu nhỏ: 20 câu, 5 câu test. Ở test, chênh lệch một câu là 20%. Kết luận chỉ đúng cho corpus
  này.
- Tự chấm: nhãn coherence và grounding do tác giả gán, và tác giả đã biết kết quả tự động trước khi
  chấm, nên có rủi ro thiên lệch xác nhận. Cách giảm thiểu là cố định vị trí mẫu trước khi chấm và
  ghi lý do từng nhãn; cách này không loại bỏ được thiên lệch.
- Model yếu so với yêu cầu: bge-small-en 384 chiều không tách được 14 abstract cùng chủ đề. 5 đến 6
  trong 20 câu có evidence không vào top-20 dù trang gold có trong top-20.
- Corpus và model đều là tiếng Anh; chưa kiểm chứng hành vi với tiếng Việt.
- Đếm trang thận trọng: 264 đã trừ hai trang checklist và supplementary bị loại nhầm, nên số thật
  cao hơn.
- Score có thể lệch ở chữ số thập phân thứ 6 giữa hai lần chạy. Khi chạy lại benchmark bằng
  `--skip-ingest`, thứ hạng và hit của cả 20 câu không đổi, chỉ score lệch ở mức đó (ví dụ 0,809681
  thành 0,80968).
- Thời gian: 1.326 s cho toàn bộ run, trong đó 1.278 s là embedding, đã tính cả lần nạp lại để kiểm
  tra idempotency. Chi phí 0 USD.

### 4.6 Nếu làm lại

1. Bỏ block tiêu đề và tác giả khỏi chunk, hoặc gộp vào abstract. Việc này sửa 5 chunk rác của B và
   giảm nhiễu vector ở trang 1.
2. Chunk theo section xuyên trang và chấm hit theo khoảng trang, để sửa lỗi q06.
3. Thử hybrid dense và BM25. Các câu miss đều hỏi về số liệu hoặc tên riêng ("19 datasets",
   "UCRL-M", "8 times"), nơi từ khóa mạnh hơn embedding.
4. Tăng benchmark lên 40 đến 50 câu với 15 câu test để thu hẹp khoảng tin cậy, và thêm câu hỏi mà
   filter là bắt buộc để đo tác động của C rõ hơn (hiện chỉ có 3 câu).

## 5. Công cụ AI và cách kiểm chứng

Bài làm được thực hiện với hỗ trợ của agent lập trình (pi). Nguyên tắc: mọi con số trong báo cáo do
script sinh ra và chạy lại được; không có số liệu, nhãn hay log nào được viết tay.

Cách kiểm chứng đã dùng:

- Evidence của cả 20 câu được validator kiểm tra tự động là substring thật trên trang gold
  (`results/benchmark_validation.json`), không dựa vào đọc bằng mắt.
- Quy tắc đếm trang được kiểm chứng bằng script in ra từng trang bị loại. Hai giả thuyết ban đầu bị
  dữ liệu bác bỏ và đã bị thay (mục 1.1).
- Metadata được kiểm tra bằng máy: text của chunk phải nằm trên đúng trang mà metadata khai báo
  (45/45).
- Nhãn coherence và grounding do tôi gán thủ công trên mẫu in ra từ `results/audit_*.csv`; lý do
  từng nhãn nằm trong `starter/scripts/apply_audit_labels.py`.
- Lỗi provider (Mistral key hết hạn) được ghi lại và xử lý bằng model local, không fallback ngẫu
  nhiên.

## 6. Bảng đối chiếu artifact

| Nội dung | Đường dẫn |
|---|---|
| Manifest corpus và quy tắc trang | `data/manifest.json` |
| Text chuẩn hóa từng trang | `data/pages.jsonl` |
| Script tải và xác minh license | `starter/scripts/fetch_corpus.py` |
| Pipeline | `starter/src/{ingestion,chunking,embeddings,store,benchmark,evaluation,run}.py` |
| Câu hỏi và gold | `benchmark/questions.jsonl` |
| Kiểm chứng benchmark | `results/benchmark_validation.json` |
| Trace top-3 từng cấu hình | `results/traces_{A,B,C}.jsonl` |
| Chỉ số | `results/metrics.json` |
| Chẩn đoán độ sâu | `results/retrieval_depth.json` |
| Nhãn thủ công | `results/audit_{coherence,grounding,metadata}.csv`, `results/audit_summary.json` |
| Nhật ký thí nghiệm | `artifacts/experiment_log.csv` |
| Cấu hình, thời gian, chi phí | `results/run_meta.json`, `results/chunk_stats.json` |
