# Two-engine architecture contract

Big-FISH = classical baseline / high-recall candidate generator. Spotiflow = independent learned detector.
Neither is ground truth. Agreement is NOT truth. Humans resolve disagreements. Raw output is kept forever.
Default engine stays `bigfish` (current behaviour, Spotiflow never imported). The default Big-FISH threshold is NOT changed.

```
image page/channel (original dtype, never 8-bit)
   ├─ Big-FISH (independent) ──► RawDetection[bigfish]  ─┐
   └─ Spotiflow (independent) ─► RawDetection[spotiflow] ─┤
                                                          ▼
                       deterministic one-to-one spatial matching (tolerance px)
                                                          ▼
        candidates: agreement | bigfish_only | spotiflow_only   (+ human_added | human_rejected from review)
                                                          ▼
              existing cross-channel association → DetectedObject (red/green/yellow/uncertain)
                                                          ▼
   human review (disagreements first, agreement QC sample)  →  audit history
                                                          ▼
          explicit dataset snapshot → grouped split → fine-tune → locked-test evaluation → explicit promotion
```

Schemas (schemas.py, already added): `EngineName, EngineChoice, ConsensusStatus, SpotiflowParams, EngineRunInfo, RawDetection,
ConsensusSummary`, `ChannelEvidence.{consensus_status,bigfish_*,spotiflow_*,match_distance_px}`,
`DetectionRequest.{engine,spotiflow,consensus_tolerance_px}`, `ReviewReason.{bigfish_only,spotiflow_only,qc_sample}`.
Raw score wording everywhere (UI, CSV, docs): **"model score"** (Spotiflow) / **"LoG response (noise units)"** (Big-FISH). Never "confidence %".

## Python interfaces
```python
# engines/base.py
@dataclass class EngineResult: run: EngineRunInfo; detections: list[RawDetection]
class EngineError(RuntimeError); class EngineUnavailable(EngineError)
class DetectionEngine(Protocol):
    name: EngineName
    def available(self) -> tuple[bool, str]            # (ok, human reason)
    def version(self) -> str
    def detect_page(self, image: np.ndarray,            # 2-D, ORIGINAL dtype, full resolution
                    *, analysis_id: str, page: int, channel_role: str|None,
                    valid_mask: np.ndarray|None, params: dict, image_sha256: dict[str,str],
                    progress: Callable[[float,str],None], cancelled: Callable[[],bool]) -> EngineResult
# engines/spotiflow_adapter.py: lazy import of spotiflow/torch (module import must succeed without torch),
#   ensure_model(model_name, cache_dir, progress) -> ModelInfo(name, version, checksum_sha256, path, cached: bool)
#   cache dir: env POLONY_MODEL_DIR or <POLONY_HOME>/models/spotiflow (download once, record sha256)
# consensus/matching.py
def match_engines(bigfish: list[RawDetection], spotiflow: list[RawDetection], tolerance_px: float) -> list[ConsensusPair]
@dataclass class ConsensusPair: status: ConsensusStatus; bigfish: RawDetection|None; spotiflow: RawDetection|None;
                                distance_px: float|None; ambiguous_with: list[str]
# detection/pipeline.py DetectionResult gains: engine_runs: list[EngineRunInfo], raw_detections: list[RawDetection]
# store (both SQLite & Postgres): save_engine_results(aid, runs, detections) (replace per run_id; append-only per rerun),
#   get_engine_runs(aid) -> list[EngineRunInfo], get_raw_detections(aid, engine=None, page=None, run_id=None, valid=None) -> list[RawDetection]
```
## REST additions (api/main.py)
- `GET /api/engines` → `[{name, available, reason, version, device:{cuda,mps,cpu}, models:[{name,cached,version,checksum}]}]`
- `POST /api/engines/spotiflow/prepare` `{model_name}` → Job (download/load with progress; cached; never re-downloads)
- `GET /api/analyses/{id}/engine-runs` → `EngineRunInfo[]`; `GET /api/analyses/{id}/raw-detections?engine=&page=&run_id=&valid=&limit=` → `RawDetection[]`
- `GET /api/analyses/{id}/consensus/summary` → `{pages:[ConsensusSummary], total:ConsensusSummary}` (per page/channel, never merged silently)
- `POST /api/analyses/{id}/consensus/qc-sample` `{n=40, seed, fraction?}` → `{marked, newly_marked, eligible, n, n_requested, population_agreements, qc:{sample_size, reviewed, confirmed, rejected, error_rate, ci95, ...}, note}` (a fixed-size deterministic random sample of unreviewed agreement objects gets review reason `qc_sample`; `fraction` maps to `max(40, fraction*N)`); `GET /consensus/summary` also returns `qc`
- detect with `engine:"hybrid"`: failure of Spotiflow ⇒ job still succeeds with Big-FISH result + failed EngineRunInfo(status failed, clear error) + warning; never an empty silent result.
- Review mapping: reject candidate = existing `delete` (tombstone ⇒ derived status human_rejected), approve = existing `confirm`, add = existing `add` (consensus_status human_added on its channel evidence).
- Review queue: engine disagreements (`bigfish_only`,`spotiflow_only`) rank above agreement; unflagged agreements not queued unless `qc_sample`.
- Exports: objects.csv per channel `<red|green>_consensus_status, _bigfish_x,_bigfish_y,_bigfish_loG_noise_units, _spotiflow_x,_spotiflow_y,_spotiflow_model_score, _engine_match_distance_px`; bundle adds `raw_detections.csv` + `engine_runs.json`; provenance lists model name/version/checksum, params, preprocessing, device.
## Dataset / ML — package `src/polony_app/ml/`, CLI `python -m polony_app.ml ...`
`dataset build|split|train|evaluate|compare|promote|rollback|list`. Snapshot = immutable content-addressed dir with `manifest.json`
(version, created, app/engine versions, source analyses + sha256, reviewer identity, coordinate convention, channel mapping used
(only when confirmed, else null), valid/excluded regions, label counts, warnings) and `spotiflow/{train,val,test}/<id>.tif + <id>.csv` (columns `y,x`; original dtype, single page, no 8-bit conversion). Positives = human-confirmed;
human-rejected/unresolved/unreviewed are listed separately and NEVER positives. Only rectangular fully-reviewed regions (fields) become crops (others skipped with warning).
Splits: group by experiment/specimen → slide → source scan; warn when independence cannot be guaranteed (reuse `legacy_import.splits`); locked test set never absorbs new corrections or trains.
Registry `<data_dir>/models/registry.json`: id, kind pretrained|finetuned, base, dataset_version, split membership, hyperparameters, seed, duration, hardware, metrics, weights sha256, status candidate|active|retired, promoted_at, previous_active. Explicit promote/rollback only; no auto-training.
Evaluation configs: bigfish default, bigfish sensitive (threshold_scale 0.5), spotiflow pretrained, spotiflow fine-tuned, agreement subset, union before review, final human-reviewed; metrics per slide/field/page/radius/model; weak/sparse results labelled (esp. green).
## Unconfirmed science (never hardcode)
TIFF page → colour is only a SUGGESTION (`ChannelInfo.suggested_role`); detection per page uses page index, colour only from the confirmed mapping.
