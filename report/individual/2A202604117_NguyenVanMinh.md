# Báo cáo cá nhân — Day 6 · Data Ingestion Engineering

**Nguyễn Văn Minh · 2A202604117 · L3B**

Mọi con số trong báo cáo này lấy từ file trong `results/` do pipeline sinh ra, không có số
liệu tự nhập. Lệnh tái lập ở mục 1.5.

## 1. Dữ liệu và pipeline

### 1.1 Chủ đề và nguồn

Corpus: **14 preprint arXiv về reinforcement learning và điều khiển tối ưu**, tải từ arXiv,
mỗi tài liệu được kiểm tra giấy phép **CC BY 4.0** trên trang abstract trước khi tải
(`starter/scripts/fetch_corpus.py` dừng ngay nếu một tài liệu không phải CC BY 4.0).

| Hạng mục | Giá trị |
|---|---|
| Số tài liệu | 14 (`data/manifest.json`) |
| Trang vật lý | 328 |
| **Trang nội dung** | **264** |
| Ký tự nội dung | 803.977 |
| Năm (theo mã arXiv) | 2020, 2021, 2022, 2023, 2024, 2025 |
| Nhóm | cs.LG (10), eess.SY (2), math.OC (1), stat.ML (1) |

Manifest ghi cho từng tài liệu: `source_id`, `title`, `source_path_or_url`, `pdf_url`,
`license_or_permission`, `accessed_at`, `primary_category`, `year`, `document_type`,
`language`, `physical_pages`, danh sách `content_pages`, `content_chars`, `sha256_pdf`,
`dropped_running_headers`, và `page_rule`.

**Quy tắc tính trang** (ghi trong manifest, từng tài liệu):
trang nội dung = trang vật lý có **≥ 1.000 ký tự** text đã chuẩn hóa, **trừ khối bibliography**
(từ heading References đến heading appendix/acknowledgement kế tiếp) và trừ trang trắng/chỉ có hình.

Quy tắc này được kiểm chứng bằng mắt, không chỉ bằng niềm tin: `starter/scripts/audit_page_rules.py`
in ra mọi trang bị loại kèm dòng đầu của trang. Hai vòng sửa trước đó đã bị loại vì dữ liệu
phản bác:
- Cắt mọi trang sau heading References → mất toàn bộ appendix (proof, hyperparameter) của
  `2206.06841v1` (12 trang còn 4). Đã sửa thành "References → heading appendix kế tiếp".
- Phân loại bibliography theo tỉ lệ dòng khớp mẫu citation → `starter/scripts/calibrate_bib_rule.py`
  cho thấy trang reference thật chỉ đạt tỉ lệ 0,02–0,50 vì entry bị wrap nhiều dòng, trong khi
  trang proof bị chấm nhầm. Bỏ hướng này.

**Lỗi và giới hạn đã biết:**
- PDF 2 cột + công thức: text bị xé theo layout block (ví dụ `xM` / `r` tách dòng). Không sửa
  được bằng text extraction; ghi nhận là giới hạn của corpus, không phải của chunker.
- Loại nhầm theo hướng thận trọng: `2410.14069v1` p15–16 (NeurIPS Paper Checklist) và
  `2506.17155v2` p17–18 (Supplementary Materials) bị tính là bibliography. Tổng cộng **264**
  trang là con số thận trọng, không phải con số phồng.
- Không có trang scan; không cần OCR.

### 1.2 Chuẩn hóa

`normalize_text` (dùng chung ở bước extract và ở pipeline): NFKC, bỏ soft hyphen, nối từ bị
gạch nối cuối dòng (`con-\nstraint` → `constraint`), gộp khoảng trắng trong dòng, giữ dòng trống
làm ranh giới đoạn. `strip_running_headers` bỏ dòng lặp trên ≥ 50% số trang (header/footer/số trang).

Một hệ quả cần biết khi đọc evidence: `real-\nworld` thành `realworld`. Câu hỏi `q03` vì vậy
dùng đúng dạng `realworld` — evidence được kiểm chứng bằng máy nên lỗi này bị bắt ngay, không
lọt vào báo cáo.

**Extract theo layout block** (`page.get_text("blocks")`) chứ không theo text phẳng: text phẳng
của bài 2 cột không có dòng trống giữa đoạn, nên cả trang thành một "đoạn" và toàn bộ heading
bị mất (lần chạy đầu chỉ phát hiện 11 section trên 264 trang). Sau khi chuyển sang block:
**229 section** ở cấu hình A và **255** ở B/C.

### 1.3 Embedding, vector store, metadata

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| Embedding | `BAAI/bge-small-en-v1.5`, 384 chiều, fastembed 0.9.1 (ONNX, local) | document dùng `embed`, query dùng `query_embed` (prefix retrieval của model) |
| Vector store | `qdrant-client` 1.20.0, embedded persistent (`qdrant_data/`), cosine | local mode = tìm kiếm chính xác (brute force), không approximate |
| Top-k | 3 | |
| Chi phí | **0 USD** | model chạy local, không gọi API |
| Thời gian | ingest 3 cấu hình 651 s; toàn bộ run 1.326 s (gồm lần nạp lại kiểm tra idempotency) | `results/run_meta.json` |

**Đổi provider — ghi rõ lý do:** lựa chọn đầu tiên là API Mistral (`mistral-embed`, 1.024 chiều).
Key sẵn có trả về `HTTP 401: Your API key expired on 2026-11-14`, nên pipeline **dừng và báo lỗi**
(`starter/src/embeddings.py` raise, không có fallback ngẫu nhiên), sau đó chuyển sang model local.
Không có vector mock ở bất kỳ bước nào.

**Metadata mỗi chunk:** `document_id`, `source`, `page`, `section`, `year`, `primary_category`,
`document_type`, `language`, `chunk_index`, `char_len`.
Trường lọc bổ sung: **`year`** (6 giá trị, lấy từ mã arXiv — không suy đoán ngày hiệu lực) và
**`primary_category`** (4 giá trị, lấy từ metadata arXiv). `section` dùng để truy vết, không dùng
để lọc vì heading do regex phát hiện nên có thể sai.

**Nạp lại không nhân đôi:** point ID = UUID5 của `chunk_id` (hash ổn định của strategy + document
+ page + index + text). Chunk cũ không còn tồn tại bị xoá theo `(strategy, document_id)`.
Kiểm chứng: nạp lại toàn bộ 3 cấu hình → **3.205 → 3.205 point** (`run_meta.idempotency.stable = true`).

### 1.4 Ba cấu hình

| | Chunk | Metadata / truy xuất |
|---|---|---|
| **A** | cửa sổ cố định 800 ký tự, overlap 150, trong từng trang | đủ metadata, **không lọc** |
| **B** | gom đoạn theo layout block + heading, tối đa 1.200 ký tự, không cắt qua heading, trong từng trang | đủ metadata, **không lọc** |
| **C** | **giống hệt B** | **thêm lọc** `year` / `primary_category` khi câu hỏi cung cấp |

A–B chỉ đổi cách chunk; B–C chỉ đổi lọc. C dùng lại đúng chunk của B nhưng được nạp riêng để
trace của C là độc lập (949 chunk mỗi cấu hình).

Chunk không vượt ranh giới trang. Đây là đánh đổi có chủ ý: `page` metadata chính xác tuyệt đối
(để chấm hit theo trang gold), nhưng câu vắt qua hai trang bị cắt. Hệ quả định lượng ở mục 4.

### 1.5 Lệnh chạy

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r starter/requirements.txt
python starter/scripts/fetch_corpus.py            # tải 14 PDF, xác minh license, sinh manifest + pages.jsonl
cd starter && ../.venv/Scripts/python.exe -m src.run --recreate
../.venv/Scripts/python.exe scripts/retrieval_depth.py
../.venv/Scripts/python.exe scripts/build_audit_samples.py
../.venv/Scripts/python.exe scripts/apply_audit_labels.py
```

`data/corpus/*.pdf` không commit (20 MB, tải lại được); `data/pages.jsonl` (text đã chuẩn hóa,
2,1 MB) được commit để người chấm chấm lại evidence mà không cần tải PDF.

## 2. Cấu hình và benchmark

**20 câu** trong `benchmark/questions.jsonl`: **15 dev / 5 test**, chia **trước khi chạy**.
Loại câu: 11 trực tiếp, 4 diễn đạt lại, 2 điều kiện/ngoại lệ, 3 câu metadata filter
(2 dev + **1 test**).

Mỗi câu có `id`, `split`, `type`, `question`, `expected_answer`, `gold_source_id`, `gold_pages`,
`evidence_quote`, `filter`, `notes`.

**Kiểm chứng evidence bằng máy, trước khi chạy:** `starter/src/benchmark.py` kiểm tra
(a) đủ trường, (b) trang gold là trang nội dung, (c) **`evidence_quote` là substring thật trên
trang gold** sau chuẩn hóa khoảng trắng, (d) giá trị filter tồn tại trong corpus, (e) đủ số câu
theo từng split và đủ câu filter. Kết quả: `results/benchmark_validation.json` → `ok: true`,
0 lỗi. Nếu evidence không khớp, pipeline dừng trước khi truy xuất.

Filter là **input do câu hỏi cung cấp trước khi truy xuất** (ví dụ "In the 2024 paper…" →
`year=2024`), không suy ra từ gold source.

**Tiêu chí chọn cấu hình (chốt trước khi so sánh):** ưu tiên (1) Evidence Hit@3 trên **dev**,
(2) grounding thủ công, (3) chi phí/tài nguyên.

**Công bố trung thực:** tôi chạy cả dev và test trong cùng một lần, nên đã thấy kết quả test
cùng lúc với dev. Vì vậy **không coi tập test là holdout độc lập**; nó là mẫu kiểm tra nhỏ
(n=5) và không dùng để tinh chỉnh tham số nào.

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

Nguồn: `results/metrics.json`, trace đầy đủ trong `results/traces_{A,B,C}.jsonl`.

### 3.2 Bốn khía cạnh còn lại

| Strategy | Coherence trung bình /2 (n=10) | Metadata đủ và đúng (n/N) | Filter tuân thủ | Grounding 0/1/2 (n=5) và trung bình |
|---|---|---|---|---|
| A | **1,0** (10×1) | 15/15 | 2/3 (dev 1/2, test 1/1) | 0×0, 1×1, 2×4 → **1,8** |
| B | **1,5** (5×2, 5×1) | 15/15 | 2/3 (dev 1/2, test 1/1) | 0×0, 1×1, 2×4 → **1,8** |
| C | 1,5 (giống B) | 15/15 | **3/3** (dev 2/2, test 1/1) | 0×0, 1×1, 2×4 → **1,8** |

- **Coherence:** 10 vị trí cố định trước khi chấm = trang 1 của 10 tài liệu đầu theo `source_id`;
  mỗi vị trí chấm chunk chứa đầu trang đó. Nhãn + lý do: `results/audit_coherence.csv`.
- **Metadata:** kiểm tra bằng máy, không bằng mắt: với mỗi bản ghi top-3 của 5 câu test (15 bản ghi
  mỗi cấu hình), script xác nhận `source`, `year`, `primary_category` khớp manifest **và** 200 ký tự
  đầu của chunk **thật sự nằm trên trang mà metadata khai báo** → 45/45 bản ghi đúng
  (`results/audit_metadata.csv`).
- **Grounding:** chấm thủ công top-3 của 5 câu test cho từng cấu hình (`results/audit_grounding.csv`).
  Người chấm là **tác giả** — rủi ro thiên lệch được nêu ở mục 4.
- Ảnh hưởng của filter (câu có filter): A và B vi phạm 1/2 câu dev (q14 lấy về chunk `year=2025`),
  C đúng 2/2. Ở test (q19) cả ba đều tình cờ đúng vì top-3 đều thuộc tài liệu 2025.

### 3.3 Chẩn đoán độ sâu truy xuất

`results/retrieval_depth.json` — tìm top-20 thay vì top-3:

| Strategy | Evidence ở top-1 | top-3 | top-10 | top-20 | Trang gold ở top-20 |
|---|---|---|---|---|---|
| A | 7 | 12 | 14 | 14 | 17 |
| B | 9 | 13 | 14 | 15 | 17 |
| C | 10 | 13 | 14 | 15 | 17 |

Đọc bảng này mới thấy bản chất của các câu miss: với 17/20 câu, **trang gold vẫn nằm trong top-20**
nhưng chunk chứa đúng câu evidence thì không. Tức là phần lớn lỗi **không phải do chunking sai**
mà do model xếp hạng kém trên một corpus rất đồng nhất (14 bài cùng chủ đề, abstract nào cũng
nói về offline RL).

`starter/scripts/explain_misses.py` kiểm tra riêng từng câu miss trên trang gold:

| Câu | Kết luận |
|---|---|
| q01, q04, q07, q09, q10 | evidence **nằm nguyên trong một chunk** nhưng bị xếp dưới (ví dụ q01: cosine 0,687 so với top-1 0,791) |
| q06 (cấu hình B) | trang gold có 3 chunk nhưng **không chunk nào chứa trọn câu evidence** — lỗi chunking thật |

## 4. Phân tích và kết luận

### 4.1 A–B: đổi cách chunk

Trên dev, B hơn A ở cả hai chỉ số (Hit@1 5→7, Hit@3 8→10) và coherence 1,0→1,5. Lý do đọc được
từ dữ liệu: cửa sổ 800 ký tự của A bắt đầu ngay ở tiêu đề/tác giả nên phần lớn chunk là
"tiêu đề + tác giả + nửa đầu abstract" — hiểu được nhưng ý bị cụt, và tiêu đề/tác giả làm loãng
vector. B gom nguyên abstract nên 5/10 vị trí đạt 2 điểm.

Nhưng B **không** thắng ở mọi mặt:
- **Test Hit@3 lại thấp hơn A** (3/5 so với 4/5). Chênh lệch đúng 1 câu: q20. Ở A, evidence của
  q20 ở rank 3; ở B/C tụt xuống rank 4. Với n=5, đây là khác biệt 1 câu — **không đủ để kết luận
  A tốt hơn B**, và tôi không dùng nó để đổi lựa chọn.
- B sinh ra **5 chunk không chứa câu nào** (chỉ tiêu đề + tác giả + địa chỉ + chữ "Abstract") trên
  14 tài liệu — `2104.10383v1` (154 ký tự), `2303.13117v1`, `2305.03360v1`, `2308.14897v1`,
  `2505.22866v1` — vì abstract dài hơn 1.200 ký tự nên không gộp được vào block tiêu đề
  (`starter/scripts/count_title_only.py`). Đây là lỗi thiết kế của B, có cách sửa rõ ràng (mục 4.6).

Đọc theo độ sâu (`results/retrieval_depth.json`) thì A→B đổi chỗ đúng ba câu: **q10** từ "không có
trong top-20" lên rank 1, **q14** từ "không có trong top-20" lên rank 2, và **q20** tụt từ 3 xuống 4.
B thắng hai câu, thua một câu — không phải thắng đều.

### 4.2 B–C: đổi lọc metadata

B–C dùng **cùng chunk**, nên mọi khác biệt đến từ filter. Kết quả: Hit@3 không đổi (10/15 dev),
Hit@1 dev 7→8, và độ tuân thủ filter 2/3→3/3. Tác động nhỏ vì chỉ 3/20 câu có filter — đúng như
dự đoán, nhưng đủ để thấy filter **không cải thiện khả năng tìm evidence, chỉ cải thiện thứ hạng
và độ đúng phạm vi**: ở B, rank 1 của q14 là một chunk `year=2025` (`2505.22866v1` p32) trong khi
chunk đúng nằm ở rank 2; ở C chunk sai phạm vi bị loại nên rank 1 là đúng trang gold.

### 4.3 Hai ví dụ sai cụ thể

1. **q06 (B) — lỗi chunking.** Câu hỏi: regret của thuật toán admission control scale thế nào
   theo số job tối đa. Trang gold `2308.02391v1` p1 có 3 chunk nhưng **không chunk nào chứa trọn
   câu evidence** (câu bị cắt qua ranh giới layout block); top-3 trả về p1, p19, p18 của cùng tài
   liệu. Đây là lỗi thật của cách gom đoạn, không phải của model. Ở A, q06 cũng miss nhưng vì lý
   do khác: evidence nguyên vẹn trong một chunk ở **rank 7**, chỉ ngoài top-3.
2. **q01 (A) — lỗi xếp hạng.** Evidence nguyên vẹn trong một chunk của `2004.01625v2` p1, nhưng
   top-3 toàn bộ là `2506.17155v2` (một bài khác). Cosine của chunk đúng 0,687 so với 0,791 của
   top-1: model không phân biệt được abstract của hai bài cùng chủ đề. Tăng top-k không cứu được
   (evidence không có trong top-20).

### 4.4 Chiến lược chọn và đánh đổi

**Chọn B làm mặc định, dùng C khi người dùng cung cấp được phạm vi (năm/nhóm).** Căn cứ: tiêu chí
đã chốt là Hit@3 trên dev → B/C = 10/15 so với A = 8/15; coherence B cao hơn (1,5 so với 1,0);
số chunk ít hơn 27 % (949 so với 1.307) nên rẻ hơn khi ingest và khi mở rộng. C không thay B ở
Hit@3 nhưng không mất gì và tăng độ đúng phạm vi, nên dùng C khi có filter.

Đánh đổi phải trả:
- Chunk theo trang → câu vắt trang bị cắt (q06). Chunk xuyên trang sẽ sửa được nhưng phải đổi
  cách chấm hit (page thành page range).
- Chunk dài hơn (838 so với 729 ký tự trung bình) → tốn context hơn khi đưa vào LLM.
- Grounding **không phân biệt được ba cấu hình** (đều 1,8/2, n=5). Trên mẫu 5 câu, khác biệt
  Hit@3 giữa các cấu hình không chuyển thành khác biệt grounding: ví dụ q20 ở B không khớp
  evidence nguyên văn nhưng top-1 vẫn nêu đúng "sample complexity bounds under latent pushforward
  coverability (Theorem 3.2)" nên vẫn đủ trả lời. **Đây là bằng chứng định lượng rằng Evidence
  Hit@3 là proxy chứ không phải thước đo grounding.**

### 4.5 Hạn chế

- **Cỡ mẫu nhỏ:** 20 câu, test 5 câu. Chênh lệch 1 câu = 20 % ở test. Không tuyên bố cấu hình nào
  "tốt nhất" nói chung; kết luận chỉ đúng cho corpus này.
- **Tự chấm:** nhãn coherence/grounding do chính tác giả gán, đã biết kết quả tự động trước khi
  chấm → rủi ro thiên lệch xác nhận. Giảm thiểu bằng cách cố định vị trí mẫu trước khi chấm và
  ghi lý do từng nhãn, nhưng không loại bỏ được.
- **Model yếu so với yêu cầu:** bge-small-en (384 chiều) không tách được 14 abstract cùng chủ đề;
  5–6/20 câu evidence không vào top-20 dù trang gold có trong top-20. Đổi sang model lớn hơn
  hoặc hybrid BM25 là hướng rõ ràng nhất, chưa làm.
- **Corpus tiếng Anh**, model tiếng Anh; không kiểm chứng được hành vi với tiếng Việt.
- **Đếm trang thận trọng:** 264 trang đã trừ 2 trang checklist/supplementary bị loại nhầm; con số
  thật cao hơn, không thấp hơn.
- **Thời gian:** 1.326 s cho toàn bộ run, trong đó 1.278 s là embedding (đã tính cả lần nạp lại
  kiểm tra idempotency, tức là làm 2 lần). Chi phí 0 USD.

### 4.6 Nếu làm lại

1. Bỏ block tiêu đề/tác giả khỏi chunk (hoặc gộp vào abstract) — sửa 5 chunk rác của B và giảm
   nhiễu vector ở trang 1.
2. Chunk theo section xuyên trang, hit chấm theo page range — sửa lỗi q06.
3. Hybrid dense + BM25: các câu miss đều là câu hỏi về số liệu/tên riêng ("19 datasets",
   "UCRL-M", "8 times"), nơi từ khóa mạnh hơn embedding.
4. Tăng benchmark lên 40–50 câu và 15 test để khoảng tin cậy hẹp lại; thêm câu hỏi mà filter là
   **bắt buộc** để đo tác động của C rõ hơn (hiện chỉ 3 câu).

## 5. Công cụ AI và cách kiểm chứng

Bài làm được thực hiện với sự hỗ trợ của agent lập trình (pi). Nguyên tắc: **mọi con số trong báo
cáo do script sinh ra và có thể chạy lại**; không có số liệu, nhãn hay log nào được viết tay.

Cách kiểm chứng đã dùng:
- Evidence của cả 20 câu được **validator kiểm tra tự động** là substring thật trên trang gold
  (`results/benchmark_validation.json`), không dựa vào việc đọc bằng mắt.
- Quy tắc đếm trang được kiểm chứng bằng script in ra từng trang bị loại; hai giả thuyết ban đầu
  bị **dữ liệu bác bỏ** và đã bị thay (mục 1.1).
- Metadata được kiểm tra bằng máy: text của chunk phải nằm trên đúng trang mà metadata khai báo
  (45/45).
- Nhãn coherence và grounding do tôi gán thủ công trên mẫu in ra từ `results/audit_*.csv`; lý do
  từng nhãn nằm trong `starter/scripts/apply_audit_labels.py`.
- Lỗi provider (Mistral key hết hạn) được ghi lại và xử lý bằng model local, không fallback ngẫu nhiên.

## 6. Bảng đối chiếu artifact

| Nội dung | Đường dẫn |
|---|---|
| Manifest corpus + quy tắc trang | `data/manifest.json` |
| Text chuẩn hóa từng trang | `data/pages.jsonl` |
| Script tải + xác minh license | `starter/scripts/fetch_corpus.py` |
| Pipeline | `starter/src/{ingestion,chunking,embeddings,store,benchmark,evaluation,run}.py` |
| Câu hỏi + gold | `benchmark/questions.jsonl` |
| Kiểm chứng benchmark | `results/benchmark_validation.json` |
| Trace top-3 từng cấu hình | `results/traces_{A,B,C}.jsonl` |
| Chỉ số | `results/metrics.json` |
| Chẩn đoán độ sâu | `results/retrieval_depth.json` |
| Nhãn thủ công | `results/audit_{coherence,grounding,metadata}.csv`, `results/audit_summary.json` |
| Nhật ký thí nghiệm | `artifacts/experiment_log.csv` |
| Cấu hình + thời gian + chi phí | `results/run_meta.json`, `results/chunk_stats.json` |
