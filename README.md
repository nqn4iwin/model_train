# model_train

공공문서 개정 해석 과제로 KORMo-10B를 학습시키고, **무엇이 학습되고 무엇이 안 되는지
지도를 그리는** 저장소다. 성능을 올리는 것이 1차 목표가 아니다.

데이터를 만드는 쪽은 `data_collect`에 있다. 그쪽은 순수 로컬이고, 이 저장소만 원격을
통해 학습 서버와 오간다.

## 저장소 구성

```text
scoring.py         정답키 없이 매기는 채점기. data_collect에서 옮겨 심은 것
test_scoring.py    옮겨 심은 채점기가 원본과 같은 답을 내는지 대조
build_records.py   교사 해석에서 쓸 것만 거르고 학습용·평가용으로 가른다
formatting.py      레코드 -> 프롬프트·정답 조립. 실험 조건이 여기서 갈린다
train.py           설정 JSON 하나로 PEFT 학습 한 번
sweep.py           설정 여러 개를 GPU 4·5에 물려 돌리고 채점까지
baseline.py        학습 전/후 KORMo를 같은 잣대로 채점
configs/           _base.json 을 물려받고 바뀌는 칸만 적는다
prompts/           역할 A 규칙서
data/              얼려둔 학습 데이터. 폴더 이름에 날짜와 판본을 박는다
docs/              베이스라인 기록, 서버 환경
```

## 학습 엔진

| 층 | 무엇 |
| --- | --- |
| 학습 루프 | **TRL `SFTTrainer`** (transformers `Trainer` 위) |
| 어댑터 | **`peft`** -- `peft_type` 이름으로 42종 중에서 고른다 |
| 모델 | `transformers` + `trust_remote_code` (KORMo는 자체 클래스) |

**한 실험은 설정 JSON 하나다.** `configs/_base.json`이 공통이고, 나머지는 `extends`로
그것을 물려받아 **바뀌는 칸만 적는다.** 비교 실험이 전부 "다른 건 똑같이 두고 하나만
바꾼다"라, 파일이 곧 그 실험의 차이 목록이 된다.

```json
{ "extends": "_base.json", "name": "lora-dora", "peft": { "use_dora": true } }
```

바꿀 수 있는 칸은 넷이다. **데이터 파일은 한 벌로 얼려두고 조건은 설정에만 둔다** --
두 실험이 정말 같은 데이터에서 출발했는지가 파일 해시 하나로 확인된다.

| 칸 | 값 | 무엇을 재나 |
| --- | --- | --- |
| `target` | `full` · `no-impacts` · `sentence` | 기획서 3.3 축 1 (무엇을 학습시키나) |
| `rules` | `true` · `false` | 규칙서 3,300자를 레코드마다 붙일지 |
| `negatives` | `keep` · `drop` | 기획서 3.3 축 2 |
| `downsample` | 숫자 · `null` | 한 계열 쏠림(60%)을 줄일지 |

## 왜 데이터를 커밋하는가

학습 데이터는 2~4MB라 커밋해도 부담이 없고, **`scp`로 올리면 서버의 파일이 어느
판본인지 아무 데도 안 적힌다.** 커밋 해시가 그 답이 된다. 비교 실험이 전부 "다른 건
똑같이 두고 하나만 바꾼다"라, 데이터가 발밑에서 바뀌면 통제가 깨진다.

모델 가중치(~20GB)는 서버에서 직접 받는다. 체크포인트 회수만 `rsync`를 쓴다.

## 채점

정답키 없이 매길 수 있는 다섯 항목이다. 정의는 `data_collect/training_data/interpret/rubric.md`에 있다.

| | |
| --- | --- |
| AM1 | 출력이 JSON 하나로 파싱된다 |
| AM2 | `대상`이 7종, `방향`이 5종 안의 말이다 |
| AM3 | 같은 `(대상, 방향)`이 두 번 안 나온다 |
| AM6s | 스스로 negative라 해놓고 `impacts`를 채우지 않았다 |
| AM8s | 자기가 낸 주체를 자기 문장에서 안 빠뜨렸다 |

**AM4·AM5·AM7은 쓸 수 없다.** 사람이 붙인 정답이 있어야 매겨지는데 원천 697건에는 없다.

**파싱이 깨지면 다섯 개가 전부 0점이다.** 그래서 평균은 사실상 파싱률을 따라간다.

## 실패 기준 — 돌리기 전에 고정된 값이다

```text
못 돌림          OOM · 예외 · 학습이 끝까지 안 감
됨               AM 다섯 평균 >= 60%  그리고  개별 최저 > 30%
이상함           돌긴 했는데 '됨'이 아닌 전부
학습 전보다 낮음   기준은 넘었는데 안 배우느니만 못한 경우
```

**결과를 보고 고치지 않는다.** 나온 것을 보고 기준을 맞추면 라운드끼리 비교가 안 된다.
스무 개 조합을 돌리면 애매한 것이 반드시 나오는데, 거기서 기준을 손대면 표 스무 개가
전부 못 쓰게 된다. `scoring.py`의 `verdict()`가 이 기준이다.

## 서버

`ad-068`, H100 80GB x8 중 **4·5번**이 이 프로젝트 몫이다. 10B를 bf16으로 올리면 20GB
정도고 LoRA는 원본을 얼려두므로 한 장에 넉넉하다. **짧은 실험 스무 개에 DDP는 통신
비용만 붙으므로, 2장이면 실험 두 개를 병렬로 돌린다.**

```bash
CUDA_VISIBLE_DEVICES=4 python train.py --config a &
CUDA_VISIBLE_DEVICES=5 python train.py --config b &
```

캐시는 저장소 밖에 둔다.

```bash
export HF_HOME="/data1/yblee/models/huggingface"
export HF_DATASETS_CACHE="/data1/yblee/datasets/huggingface"
```

## 참고

- `data_collect/docs/기획서_최종.md` 3장(학습)·4장(평가)
- `data_collect/training_data/설계_메모.md` 교사 출력 설계
- `data_collect/training_data/interpret/rubric.md` 채점 항목 정의
