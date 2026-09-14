# 재개 안내 — native coverage 실행

세션이 끊겨도 이 디렉터리만 보면 상태를 알 수 있다. 작업은 detached `screen`에서 돌므로
VSCode/SSH/네트워크가 끊겨도 계속된다.

## 상태 확인

```bash
cd "/data1/home/chlee/projects/immune virtual cell/ivcbench"
./runs/status.sh          # 전 작업의 상태 · 로그 크기 · 마지막 줄
screen -ls                # 살아있는 세션
tail -f runs/<job>.log    # 특정 작업 따라가기
```

`runs/<job>.status` 는 `RUNNING` → `DONE:0` 또는 `FAILED:<code>` 로 끝난다.
`runs/<job>.cmd` 에 실행한 명령이 그대로 들어 있다.

## 새 작업 올리기

```bash
./runs/launch.sh <job-id> "<command>"
```

## 반드시 지킬 것 — 실행 환경

**`ivcbench/.venv/bin/python` 으로만 돌린다.** `benchmark/.venv` 는 `benchmark/src` 를 가리키는
editable 설치라 `_RUNNER_DIR` 이 `benchmark/model_runners`(러너 20개, STATE 출력 선택 수정 없음)로
잡힌다. 그 경로로 돌리면 STATE 가 `adata_real.h5ad`(고정 시드 control 부분표본)를 회수해
**과거 값을 그대로 재현**하고, 새 러너들은 존재하지도 않는다. 한 라운드를 통째로 날린 뒤에야 발견했다.

`scripts/_env_guard.py` 가 `run_cluster.py`·`assemble_cross_cluster.py`·`census_units.py` 에서
이를 막는다. 새 진입점을 만들면 같은 가드를 넣는다.

`results/_invalid_wrong_venv/` 는 잘못된 환경으로 돌린 산출물이다. **삭제하지 말고, 어떤 분석에도
쓰지 않는다.**

## GPU 배치

| GPU | 작업 |
|---|---|
| 0 | scgpt_4x (10 epoch / 32k cells, 96 donors) |
| 1 | scgpt_12x (30 epoch / 32k cells, 96 donors) |
| 2 | state_C5 (T5c · T5u) |
| 3 | state_C3 (T3) |

## 끝난 뒤 할 일

1. `runs/status.sh` 로 `DONE:0` 확인. `FAILED` 면 로그의 traceback 부터 본다.
2. 새 예측이 실제로 학습 산출물인지 검증한다 — **과거 값과 동일하면 실패로 간주한다**
   (historical: `results/C3/results_raw.csv`, `results/C5/results_raw.csv`).
3. `results/_paper/withdrawn_bundles.csv` 에 새 실행이 들어가지 않았는지 확인한다.
4. 센서스 재조립 → 통계 재생성 → 표/그림 → 문서 순서로 진행한다.
   경로는 `revision_claude/CHANGE_IMPACT_TABLE.md` 의 영향표를 따른다.

## 남은 실행 (계획 §3 기준)

- **진행 중:** STATE T3·T5c·T5u (state_C3, state_C5), scGPT 예산 사다리 2점
- **다음:** STATE T4(C4), scGen T5c(`scgen_c5_loct_runner.py` 작성 완료, 미실행),
  linear-shift-KOemb T3
- **구현 필요:** CPA T1/T2/T5c/T5u(원 predictor 복원), PertAdapt T3/T4(GEARS graph 경로),
  scFoundation T3/T4(공개 GEARS integration), PerturbNet T1/T2/T5c(categorical variant),
  biolord T1/T2/T5c(`unknown_attributes=False` 재학습)
- **기존 N/L 재검증:** CellFlow 6칸, CellOT 3칸, scPRAM 3칸, GEARS·AttentionPert 각 2칸
