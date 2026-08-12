# Stage 1 — Chunk Strategy Selection

## Notebooklm

| Strategy | Precision | Recall | F1 | Correctness | Faithfulness | Retrieve time (s) | Generate time (s) | True abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed size | 0.9057 | 0.9290 | 0.8916 | 0.6214 | 0.9433 | 0.5032 | 6.1303 | 0.8936 |
| HSF | 0.9351 | 0.8964 | **0.8951** | 0.6223 | 0.9318 | **0.4370** | 6.5478 | **0.9043** |
| RC | 0.9043 | 0.9167 | 0.8881 | **0.6720** | 0.9384 | 0.4836 | 6.9727 | 0.8830 |

## Studyguide

| Strategy | Precision | Recall | F1 | Correctness | Faithfulness | Retrieve time (s) | Generate time (s) | True abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed size | 0.7670 | 0.8287 | 0.7083 | **0.4974** | 0.6866 | 0.5627 | **7.3144** | **0.5472** |
| HSF | **0.8288** | **0.8632** | **0.7868** | 0.4791 | 0.5971 | **0.5033** | 7.8360 | **0.5472** |
| RC | 0.7464 | 0.8404 | 0.7124 | 0.4618 | **0.7052** | 0.5243 | 7.8733 | 0.5132 |

## Overall comparison

| Strategy | Notebooklm F1 / Correctness | Studyguide F1 / Correctness | Summary |
|---|---:|---:|---|
| Fixed size | 0.892 / 0.621 | 0.708 / 0.497 | Faithfulness và tốc độ generation tốt |
| HSF | **0.895 / 0.622** | **0.787 / 0.479** | Retrieval mạnh nhất, đặc biệt trên studyguide |
| RC | 0.888 / **0.672** | 0.712 / 0.462 | Correctness notebooklm tốt, nhưng generation chậm nhất |

## Selection

Chọn **HSF, chunk size 600** cho Stage 2 vì đây là strategy có chất lượng retrieval tổng thể tốt nhất, với F1 cao nhất ở cả hai datasets và retrieval nhanh nhất trên cả hai.

## Stage 2 — Retrieve Strategy Evaluation

### Loop 1: HSF/600, normal vs multistage vector 0.5 / BM25 0.5

#### Notebooklm

| Retrieve strategy | Precision | Recall | F1 | Correctness | Faithfulness | Retrieve time (s) | Generate time (s) | True abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | 0.9351 | 0.8964 | 0.8951 | 0.6223 | 0.9318 | **0.4370** | 6.5478 | 0.9043 |
| Multistage 0.5/0.5 | **0.9594** | **0.9541** | **0.9467** | **0.6877** | **0.9495** | 2.4788 | **6.5252** | **0.9149** |

#### Studyguide

| Retrieve strategy | Precision | Recall | F1 | Correctness | Faithfulness | Retrieve time (s) | Generate time (s) | True abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | **0.8288** | **0.8632** | **0.7868** | **0.4791** | 0.5971 | **0.5033** | **7.8360** | **0.5472** |
| Multistage 0.5/0.5 | 0.7804 | 0.7878 | 0.6912 | 0.4562 | **0.5987** | 2.2040 | 8.0982 | 0.4415 |

### Loop 1 decision

Multistage `0.5/0.5` cải thiện đáng kể notebooklm nhưng làm giảm F1 và correctness trên studyguide, đồng thời tăng retrieve time khoảng `1.7–2.0s`. Vì vậy, Loop 2 sẽ test multistage với trọng số vector `0.75` và BM25 `0.25`: giảm ảnh hưởng BM25 để kiểm tra liệu cấu hình thiên về vector có thể giữ lợi ích của multistage trên notebooklm và cải thiện studyguide hay không.

### Final comparison: normal vs multistage 0.5/0.5 vs multistage 0.75/0.25

#### Notebooklm

| Retrieve strategy | Precision | Recall | F1 | Correctness | Faithfulness | Retrieve time (s) | Generate time (s) | True abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | 0.9351 | 0.8964 | 0.8951 | 0.6223 | 0.9318 | **0.4370** | 6.5478 | 0.9043 |
| Multistage 0.5/0.5 | **0.9594** | **0.9541** | **0.9467** | 0.6877 | **0.9495** | 2.4788 | 6.5252 | 0.9149 |
| Multistage 0.75/0.25 | 0.9583 | 0.9454 | 0.9420 | **0.7047** | 0.9193 | 6.5699 | **6.0534** | **0.9255** |

#### Studyguide

| Retrieve strategy | Precision | Recall | F1 | Correctness | Faithfulness | Retrieve time (s) | Generate time (s) | True abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | **0.8288** | **0.8632** | **0.7868** | **0.4791** | 0.5971 | **0.5033** | 7.8360 | **0.5472** |
| Multistage 0.5/0.5 | 0.7804 | 0.7878 | 0.6912 | 0.4562 | 0.5987 | 2.2040 | 8.0982 | 0.4415 |
| Multistage 0.75/0.25 | 0.8112 | 0.8091 | 0.7414 | 0.4683 | **0.6136** | 3.6360 | **7.6906** | 0.4717 |

### Overall comparison

| Retrieve strategy | Notebooklm F1 / Correctness | Studyguide F1 / Correctness | Retrieve time (notebooklm / studyguide) | Summary |
|---|---:|---:|---:|---|
| Normal | 0.895 / 0.622 | **0.787 / 0.479** | **0.44s / 0.50s** | Ổn định nhất giữa hai datasets, nhanh nhất |
| Multistage 0.5/0.5 | **0.947 / 0.688** | 0.691 / 0.456 | 2.48s / 2.20s | Tốt nhất trên notebooklm, giảm mạnh studyguide |
| Multistage 0.75/0.25 | 0.942 / **0.705** | 0.741 / 0.468 | 6.57s / 3.64s | Chất lượng tổng thể tốt nhất, đổi lại latency cao nhất |

### Stage 2 selection

Chọn **HSF + multistage retrieve (vector 0.75 / BM25 0.25)** cho Stage 3. Cấu hình này ưu tiên chất lượng tổng thể: F1 trung bình gần tương đương normal retrieve, trong khi correctness và faithfulness trung bình cao hơn trên hai datasets. Trade-off chấp nhận là retrieve latency cao hơn.

Ưu tiên chất lượng ở thời điểm này vì retrieve strategy quyết định evidence đầu vào cho quá trình sinh câu trả lời. Việc tối ưu chunk size hoặc budget token ở các stage sau chỉ có thể cải thiện một phần chất lượng, không tái tạo đầy đủ lợi ích của hybrid search và reranking. Ngược lại, latency có thể được tối ưu sau bằng các thay đổi kỹ thuật như giới hạn candidate trước reranking, caching, chạy song song các nhánh retrieve hoặc dùng reranker nhẹ hơn. Lựa chọn này giả định không có ràng buộc latency cứng.

## Stage 3 - Chunk size optimization

Configuration held constant: HSF chunking, multistage retrieval (vector 0.75 / BM25 0.25), token budget 2,000.

### Notebooklm

| Chunk size | Precision | Recall | F1 | Correctness | Faithfulness | Retrieve time (s) | Generate time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 400 | 0.9315 | 0.9232 | 0.9085 | 0.6789 | 0.9532 | **1.6766** | 8.3508 |
| 600 | 0.9583 | **0.9454** | 0.9420 | **0.7047** | 0.9193 | 6.5699 | **6.0534** |
| 800 | **0.9749** | 0.9386 | **0.9459** | 0.6292 | **0.9827** | 1.7301 | 6.4926 |

### Studyguide

| Chunk size | Precision | Recall | F1 | Correctness | Faithfulness | Retrieve time (s) | Generate time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 400 | 0.7880 | 0.8030 | 0.7205 | 0.4591 | **0.6458** | 1.7635 | 8.7896 |
| 600 | **0.8112** | **0.8091** | **0.7414** | **0.4683** | 0.6136 | 3.6360 | **7.6906** |
| 800 | 0.7666 | 0.7980 | 0.6996 | 0.4435 | 0.5955 | **1.6595** | 8.7752 |

### Stage 3 selection

Choose **HSF, multistage retrieval (vector 0.75 / BM25 0.25), chunk size 600** for Stage 4. It has the best overall F1 and correctness, leads clearly on studyguide, and has the lowest generation time on both datasets. The marginal notebooklm F1 gain at size 800 does not offset its lower answer correctness on both datasets.

## Stage 4 - Retrieve budget optimization

Configuration held constant: HSF chunking, chunk size 600, multistage retrieval (vector 0.75 / BM25 0.25).

### Notebooklm

| Token budget | Precision | Recall | F1 | Correctness | Faithfulness | Retrieve time (s) | Generate time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1,600 | 0.9639 | 0.9171 | 0.9273 | 0.6642 | 0.9524 | 2.6118 | 6.8088 |
| 2,000 | 0.9583 | 0.9454 | **0.9420** | **0.7047** | 0.9193 | 6.5699 | **6.0534** |
| 2,400 | 0.9292 | 0.9273 | 0.9092 | 0.6929 | 0.9548 | **2.2809** | 8.4463 |

### Studyguide

| Token budget | Precision | Recall | F1 | Correctness | Faithfulness | Retrieve time (s) | Generate time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1,600 | 0.7833 | **0.8233** | 0.7252 | 0.4645 | **0.6500** | 2.4127 | 8.7980 |
| 2,000 | **0.8112** | 0.8091 | **0.7414** | 0.4683 | 0.6136 | 3.6360 | **7.6906** |
| 2,400 | 0.7492 | 0.8150 | 0.7013 | **0.4710** | 0.6182 | **2.3530** | 9.1416 |

### Final selection

Choose **HSF + chunksize 600 + multistage retrieval (vector 0.75 / BM25 0.25) + token budget 2,000**. This budget achieves the highest F1 on both datasets and the highest notebooklm correctness; increasing to 2,400 does not yield a stable quality gain.
