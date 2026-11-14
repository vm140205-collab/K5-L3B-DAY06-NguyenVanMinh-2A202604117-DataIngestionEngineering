# Scripts

Chạy từ thư mục gốc repo, dùng python trong `.venv`. Không script nào cần API key.

## Đường chạy chính

| Script | Việc |
|---|---|
| `fetch_corpus.py` | Tải 14 PDF từ arXiv, xác minh license CC BY 4.0, sinh `data/manifest.json` và `data/pages.jsonl`. `--verify` chỉ kiểm tra lại license. |
| `validate_benchmark.py` | Kiểm chứng `benchmark/questions.jsonl`: evidence phải là substring thật trên trang gold, đủ split, đủ câu filter. |
| `retrieval_depth.py` | Tìm top-20 cho cả ba cấu hình, ghi `results/retrieval_depth.json`. Tách lỗi xếp hạng khỏi lỗi chunking. |
| `build_audit_samples.py` | Sinh mẫu chấm thủ công: 10 vị trí coherence, 5 câu test grounding, và kiểm tra metadata bằng máy. |
| `apply_audit_labels.py` | Ghi nhãn thủ công (kèm lý do) vào `results/audit_*.csv`, tổng hợp `results/audit_summary.json`. |
| `verify_report.py` | Đối chiếu từng con số trong báo cáo với `results/`. Đây là cổng kiểm tra cuối; fail nếu lệch. |

## Chẩn đoán và kiểm chứng

| Script | Việc |
|---|---|
| `audit_page_rules.py` | In ra mọi trang bị loại khỏi số trang nội dung, kèm dòng đầu, để kiểm tra quy tắc bằng mắt. |
| `calibrate_bib_rule.py` | Đo tỉ lệ dòng giống citation trên từng trang; dùng để bác bỏ cách phân loại bibliography theo tỉ lệ dòng. |
| `explain_misses.py` | Với mỗi câu miss: evidence nằm nguyên trong một chunk hay bị tách, và cosine của chunk đó so với top-1. |
| `count_title_only.py` | Đếm chunk trang 1 chỉ có tiêu đề và tác giả (lỗi thiết kế của cấu hình B). |
| `check_chunk_quality.py` | Đếm chunk quá ngắn, chunk thiếu nhãn section, trang nội dung không có chunk. |
| `smoke_chunking.py` | Chạy chunking không cần API/store, in thống kê và vài chunk mẫu. |
| `show_sections.py` | In các heading phát hiện được trên từng trang của một tài liệu. |
| `find_evidence.py` | Tìm câu verbatim trong corpus để viết `evidence_quote`. |
| `view_audit.py` | In `results/audit_coherence.csv` hoặc `audit_grounding.csv` để đọc lại và chấm lại nhãn. |
| `style_check.py` | Đếm dấu hiệu văn phong máy trong một file markdown (em dash, ngoặc cong, bold, độ dài câu). |
| `analyze_failures.py` | Liệt kê câu miss theo cấu hình kèm top-3, và kiểm tra tuân thủ filter. |

`check_setup.py` thuộc starter gốc (kiểm tra PostgreSQL + pgvector); bài này không dùng nhánh đó.
