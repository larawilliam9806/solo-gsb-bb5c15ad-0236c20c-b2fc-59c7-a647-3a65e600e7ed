"""跨展柜环境传播分析：时滞相关纯算法单测 + 接口集成测试（内存 SQLite）。"""

from __future__ import annotations

import datetime as dt
import math
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine, init_db
from app.main import app
from app.propagation import (
    REASON_CONSTANT,
    REASON_INSUFFICIENT,
    REASON_NO_OVERLAP,
    LagCorrelation,
    PropagationValidationError,
    compute_propagation,
)
from app.propagation import (
    MAX_GRID_POINTS,
    MAX_LAG_ROWS,
    MAX_PAIR_WORK,
    _pair_work,
)
from app.segmentation import Sample

BASE = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    init_db()
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


def _sample(minute: float, t: float, h: float = 50.0) -> Sample:
    return Sample(BASE + dt.timedelta(seconds=60 * minute), t, h)


def _reading(sn, ts, t, h):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


# 非对称伪随机波形：平移后只有正确时滞相关为 1，线性斜坡不会产生多峰退化
WAVE = [20.0, 22.0, 21.0, 25.0, 24.0, 28.0, 27.0, 30.0, 26.0, 23.0, 21.0, 24.0]


def _wave_src():
    return [_sample(i, WAVE[i]) for i in range(12)]


def _compute(src, tgt, dimension="temperature", *, start=BASE, end=None,
             step=60.0, max_lag=300.0, max_gap=600.0):
    return compute_propagation(
        src,
        tgt,
        dimension,
        start=start,
        end=end if end is not None else BASE + dt.timedelta(seconds=60 * 13),
        step_seconds=step,
        max_lag_seconds=max_lag,
        max_gap_seconds=max_gap,
    )


# ---------- 纯算法：时滞方向 ----------


def test_positive_lag_means_source_leads_target():
    # target 取值整体后移 2 分钟：source 在 t 的值 == target 在 t+120s 的值
    src = _wave_src()
    tgt = [_sample(i + 2, WAVE[i]) for i in range(12)]
    res = _compute(src, tgt)
    assert res.best_lag_seconds == 120.0
    assert res.best_correlation == pytest.approx(1.0, abs=1e-9)
    assert res.direction == "source_to_target"
    # 最佳明细行与汇总一致
    best_row = next(l for l in res.lags if l.lag_seconds == 120.0)
    assert best_row.correlation == pytest.approx(1.0, abs=1e-9)


def test_negative_lag_means_target_leads_source():
    src = [_sample(i + 2, WAVE[i]) for i in range(12)]
    tgt = _wave_src()
    res = _compute(src, tgt)
    assert res.best_lag_seconds == -120.0
    assert res.direction == "target_to_source"


def test_zero_lag_is_synchronous():
    src = _wave_src()
    res = _compute(src, src, end=BASE + dt.timedelta(seconds=60 * 11), max_lag=120.0)
    assert res.best_lag_seconds == 0.0
    assert res.direction == "synchronous"
    assert res.best_correlation == pytest.approx(1.0, abs=1e-12)


def test_humidity_dimension_uses_humidity_values():
    hum = [40.0, 44.0, 41.0, 47.0, 43.0, 50.0, 42.0, 46.0]
    src = [_sample(i, 20.0, hum[i]) for i in range(8)]
    tgt = [_sample(i + 1, 20.0, hum[i]) for i in range(8)]
    res = _compute(
        src,
        tgt,
        "humidity",
        end=BASE + dt.timedelta(seconds=60 * 8),
        max_lag=180.0,
    )
    assert res.best_lag_seconds == 60.0
    assert res.direction == "source_to_target"


# ---------- 纯算法：网格、断档、有效性 ----------


def test_grid_clipped_to_overlap_and_floored():
    src = _wave_src()  # 覆盖 0..11 分钟
    tgt = [_sample(i + 2, WAVE[i]) for i in range(12)]  # 覆盖 2..13 分钟
    res = _compute(src, tgt)
    # 重叠 [2min, 11min]：10 个 60s 网格点
    assert res.overlap_start == BASE + dt.timedelta(seconds=120)
    assert res.overlap_end == BASE + dt.timedelta(seconds=660)
    assert res.grid_point_count == 10
    assert res.source_valid_count == 10
    assert res.target_valid_count == 10


def test_grid_does_not_extend_past_end_when_step_does_not_divide_window():
    res = _compute(
        _wave_src(),
        _wave_src(),
        end=BASE + dt.timedelta(seconds=630),  # 10.5 分钟
        step=240.0,  # 0,4,8 分钟；下一个 12 分钟超出
        max_lag=0.0,
    )
    assert res.grid_point_count == 3


def test_gap_larger_than_max_gap_invalidates_interior_points():
    src = _wave_src()
    # target 在 1→11 分钟有 10 分钟断档；max_gap=300
    tgt = [_sample(0, 20.0), _sample(1, 22.0), _sample(11, 21.0), _sample(12, 25.0)]
    res = _compute(src, tgt, end=BASE + dt.timedelta(seconds=60 * 12), max_lag=0.0, max_gap=300.0)
    # 重叠裁剪到 source 覆盖 [0,11]：网格 0..11；target 仅在读数点 0,1,11 有效
    assert res.grid_point_count == 12
    assert res.target_valid_count == 3
    lag0 = res.lags[0]
    assert lag0.pair_count == 3  # 0,1,11 两侧都有效


def test_gap_equal_to_max_gap_is_still_continuous():
    samples = [_sample(0, 20.0), _sample(10, 30.0)]
    res = _compute(
        samples,
        samples,
        end=BASE + dt.timedelta(seconds=600),
        step=300.0,
        max_lag=0.0,
        max_gap=600.0,  # 恰好等于读数间隔
    )
    assert res.target_valid_count == 3
    assert res.lags[0].available is True
    assert res.lags[0].correlation == pytest.approx(1.0, abs=1e-12)


# ---------- 纯算法：不可用场景绝不产生 NaN ----------


def test_no_overlap_marks_all_lags_unavailable():
    src = _wave_src()
    far = [_sample(60, 1.0), _sample(61, 2.0)]
    res = _compute(src, far, max_lag=120.0)
    assert res.direction == "unavailable"
    assert res.overlap_start is None
    assert res.overlap_end is None
    assert res.grid_point_count == 0
    assert res.best_lag_seconds is None
    assert res.best_correlation is None
    assert res.lags
    for lag in res.lags:
        assert lag.available is False
        assert lag.correlation is None
        assert lag.reason == REASON_NO_OVERLAP


def test_constant_series_is_unavailable_not_nan():
    const = [_sample(i, 20.0) for i in range(12)]
    res = _compute(const, const, end=BASE + dt.timedelta(seconds=60 * 11), max_lag=120.0)
    assert res.direction == "unavailable"
    for lag in res.lags:
        assert lag.available is False
        assert lag.correlation is None
        assert lag.reason == REASON_CONSTANT


def test_one_side_constant_pairs_is_constant_reason():
    src = _wave_src()
    tgt = [_sample(i, 50.0) for i in range(12)]
    res = _compute(src, tgt, end=BASE + dt.timedelta(seconds=60 * 11))
    assert all(not lag.available and lag.reason == REASON_CONSTANT for lag in res.lags)
    assert res.direction == "unavailable"


def test_insufficient_pairs_when_only_one_point_valid():
    # 两侧都只有一个读数点：网格仅 1 点，无法计算相关
    src = [_sample(0, 20.0)]
    tgt = [_sample(0, 21.0)]
    res = _compute(src, tgt, end=BASE, max_lag=0.0)
    lag0 = res.lags[0]
    assert lag0.pair_count == 1
    assert lag0.available is False
    assert lag0.correlation is None
    assert lag0.reason == REASON_INSUFFICIENT
    assert res.direction == "unavailable"


def test_gap_endpoints_two_nonconstant_points_still_correlate():
    # 读数间隔 660s > 600：中间插值点无效，但两个读数端点本身仍有效 → 2 对可计算
    src = [_sample(0, 20.0), _sample(11, 24.0)]
    tgt = [_sample(0, 21.0), _sample(11, 25.0)]
    res = _compute(src, tgt, end=BASE + dt.timedelta(seconds=60 * 11), max_lag=0.0)
    lag0 = res.lags[0]
    assert lag0.pair_count == 2
    assert lag0.available is True
    assert lag0.correlation == pytest.approx(1.0)


def test_no_nan_anywhere():
    scenarios = [
        (_wave_src(), [_sample(i + 2, WAVE[i]) for i in range(12)]),
        (_wave_src(), [_sample(60, 1.0), _sample(61, 2.0)]),
        ([_sample(i, 20.0) for i in range(12)], [_sample(i, 20.0) for i in range(12)]),
    ]
    for src, tgt in scenarios:
        res = _compute(src, tgt)
        for lag in res.lags:
            assert lag.correlation is None or not math.isnan(lag.correlation)
        assert res.best_correlation is None or not math.isnan(res.best_correlation)


# ---------- 纯算法：择优次序 ----------


def test_select_best_abs_correlation_first():
    rows = [
        LagCorrelation(0.0, True, 9, 0.9, None),
        LagCorrelation(120.0, True, 4, -0.95, None),
    ]
    from app.propagation import _select_best

    assert _select_best(rows) == (120.0, -0.95)


def test_select_best_pair_count_then_lag_then_sign():
    from app.propagation import _select_best

    # |r| 相同：成对点数更多
    assert _select_best([
        LagCorrelation(0.0, True, 4, 0.9, None),
        LagCorrelation(60.0, True, 5, 0.9, None),
    ]) == (60.0, 0.9)
    # |r|、点数相同：|lag| 更小
    assert _select_best([
        LagCorrelation(-60.0, True, 5, 0.9, None),
        LagCorrelation(0.0, True, 5, 0.9, None),
        LagCorrelation(60.0, True, 5, 0.9, None),
    ]) == (0.0, 0.9)
    # 全部相同仅符号相反：时滞值更小（负者）
    assert _select_best([
        LagCorrelation(-60.0, True, 5, 0.9, None),
        LagCorrelation(60.0, True, 5, 0.9, None),
    ]) == (-60.0, 0.9)
    # 无可用
    assert _select_best([LagCorrelation(0.0, False, 1, None, REASON_INSUFFICIENT)]) == (
        None,
        None,
    )


def test_lags_enumerated_ascending_and_bounded():
    src = _wave_src()
    res = _compute(src, src, end=BASE + dt.timedelta(seconds=60 * 11), max_lag=300.0)
    lags = [lag.lag_seconds for lag in res.lags]
    assert lags == sorted(lags)
    assert lags[0] == -300.0
    assert lags[-1] == 300.0
    assert 0.0 in lags

    # max_lag < step：仅枚举零时滞
    res_small = _compute(
        src, src, end=BASE + dt.timedelta(seconds=60 * 11), step=600.0, max_lag=100.0
    )
    assert [lag.lag_seconds for lag in res_small.lags] == [0.0]


def test_unsorted_input_deterministic():
    src = _wave_src()
    tgt = [_sample(i + 2, WAVE[i]) for i in range(12)]
    r1 = _compute(src, tgt)
    r2 = _compute(list(reversed(src)), list(reversed(tgt)))
    assert [ (l.lag_seconds, l.correlation, l.pair_count) for l in r1.lags] == [
        (l.lag_seconds, l.correlation, l.pair_count) for l in r2.lags
    ]
    assert r1.best_lag_seconds == r2.best_lag_seconds


# ---------- 纯算法：微秒精度、取整与有限性 ----------


def test_sub_microsecond_positive_step_rounds_to_zero_and_raises():
    src = _wave_src()
    # 小于半微秒的正步长：微秒取整为零，不再除零
    for tiny in (1e-7, 1e-9, 4e-7):
        with pytest.raises(PropagationValidationError):
            _compute(src, src, step=tiny, max_lag=0.0)


def test_nonpositive_step_raises_validation_error():
    src = _wave_src()
    with pytest.raises(PropagationValidationError):
        _compute(src, src, step=0.0, max_lag=0.0)
    with pytest.raises(PropagationValidationError):
        _compute(src, src, step=-1.0, max_lag=0.0)


def test_nonfinite_time_params_raise_validation_error():
    src = _wave_src()
    for step, lag, gap in [
        (float("nan"), 300.0, 600.0),
        (float("inf"), 300.0, 600.0),
        (60.0, float("nan"), 600.0),
        (60.0, float("inf"), 600.0),
        (60.0, 300.0, float("nan")),
        (60.0, 300.0, float("inf")),
    ]:
        with pytest.raises(PropagationValidationError):
            _compute(src, src, step=step, max_lag=lag, max_gap=gap)


def test_step_normalized_to_microsecond_and_echoed_everywhere():
    # 60.0004µs → 取整为 60µs；跨度 0.59994s（9999 个间隔，恰好 10000 点边界）。
    # 仅两端读数 + 极大 max_gap，使全部网格点可插值。
    end = BASE + dt.timedelta(microseconds=599_940)
    src = [Sample(BASE, 20.0, 50.0), Sample(end, 25.0, 50.0)]
    tgt = [Sample(BASE, 21.0, 50.0), Sample(end, 26.0, 50.0)]
    res = _compute(src, tgt, end=end, step=60.0004e-6, max_lag=180e-6,
                   max_gap=1e12)
    assert res.step_seconds == 60e-6
    assert res.grid_point_count == MAX_GRID_POINTS
    # 时滞秒数是生效步长的整数倍，最大档位不超过 max_lag
    lags = [lag.lag_seconds for lag in res.lags]
    assert lags[0] == -180e-6
    assert lags[-1] == 180e-6
    assert lags == sorted(lags)
    # 每档时滞都是生效步长的整数倍
    assert all(abs(lag_seconds / 60e-6 - round(lag_seconds / 60e-6)) < 1e-9
               for lag_seconds in lags)


def test_pair_work_formula():
    assert _pair_work(10, 2) == 10 * 5 - 2 * 3  # 44
    assert _pair_work(2, 5) == 4  # |k| ≥ n 时无配对，合计 n²
    assert _pair_work(1, 0) == 1
    # kmax < n：n=5025、kmax=100 → 999_925（差 126，下一点即超限）
    assert _pair_work(5025, 100) == MAX_PAIR_WORK - 75
    assert _pair_work(5026, 100) == MAX_PAIR_WORK + 126


def test_pair_work_exact_cap_when_kmax_covers_all_lags():
    # kmax ≥ n 时所有时滞的配对合计恰为 n²：n=1000 → 恰好 1_000_000
    assert _pair_work(1000, 999) == MAX_PAIR_WORK
    assert _pair_work(1000, 2000) == MAX_PAIR_WORK
    assert _pair_work(1001, 1001) == MAX_PAIR_WORK + 2001


# ---------- 纯算法：分配前规模上限（边界值可计算） ----------


def _boundary_samples(span: dt.timedelta) -> tuple[list[Sample], list[Sample]]:
    # 仅两端读数 + 极大 max_gap：覆盖整个窗口且每点都可插值
    return (
        [Sample(BASE, 20.0, 50.0), Sample(BASE + span, 25.0, 50.0)],
        [Sample(BASE, 21.0, 50.0), Sample(BASE + span, 26.0, 50.0)],
    )


def test_grid_point_limit_boundary_computes():
    span = dt.timedelta(seconds=MAX_GRID_POINTS - 1)  # 1s 步长 → 恰好 10000 点
    src, tgt = _boundary_samples(span)
    res = _compute(src, tgt, end=BASE + span, step=1.0, max_lag=0.0, max_gap=1e12)
    assert res.grid_point_count == MAX_GRID_POINTS


def test_grid_point_over_limit_raises():
    span = dt.timedelta(seconds=MAX_GRID_POINTS)  # 1s 步长 → 10001 点
    src, tgt = _boundary_samples(span)
    with pytest.raises(PropagationValidationError):
        _compute(src, tgt, end=BASE + span, step=1.0, max_lag=0.0, max_gap=1e12)


def test_lag_row_limit_boundary_computes():
    # kmax=100 → 201 行；max_lag=100.9 仍向下取整到 100
    src = _wave_src()
    res = _compute(src, src, end=BASE + dt.timedelta(seconds=60 * 11),
                   step=60.0, max_lag=100.9 * 60.0)
    assert len(res.lags) == MAX_LAG_ROWS


def test_lag_row_over_limit_raises_even_without_overlap():
    # 时滞行数先于重叠检查：无重叠同样拒绝无界明细
    src = _wave_src()
    far = [_sample(60, 1.0), _sample(61, 2.0)]
    with pytest.raises(PropagationValidationError):
        _compute(src, far, step=60.0, max_lag=101.0 * 60.0)


def test_pair_work_over_limit_around_cap():
    # 1s 步长、kmax=100：n=5025 时工作量 999_925（放行）；n=5026 时 1_000_126（拒绝）
    last_ok_n = 5025
    assert _pair_work(last_ok_n, 100) <= MAX_PAIR_WORK
    assert _pair_work(last_ok_n + 1, 100) > MAX_PAIR_WORK
    span_ok = dt.timedelta(seconds=last_ok_n - 1)
    src, tgt = _boundary_samples(span_ok)
    res = _compute(src, tgt, end=BASE + span_ok, step=1.0, max_lag=100.0,
                   max_gap=1e12)
    assert res.grid_point_count == last_ok_n
    assert len(res.lags) == MAX_LAG_ROWS

    span_bad = dt.timedelta(seconds=last_ok_n)
    src2, tgt2 = _boundary_samples(span_bad)
    with pytest.raises(PropagationValidationError):
        _compute(src2, tgt2, end=BASE + span_bad, step=1.0, max_lag=100.0,
                 max_gap=1e12)


# ---------- 接口集成 ----------


def _ingest(client, sn, rows):
    r = client.post("/api/v1/readings/batch", json={"readings": rows})
    assert r.status_code == 200, r.text


def _wave_readings(sn, offset_minutes=0):
    return [
        _reading(
            sn,
            (BASE + dt.timedelta(seconds=60 * (i + offset_minutes))).isoformat().replace(
                "+00:00", "Z"
            ),
            WAVE[i],
            50.0,
        )
        for i in range(12)
    ]


BASE_PARAMS = {
    "start": "2026-09-01T00:00:00Z",
    "end": "2026-09-01T00:13:00Z",
    "step_seconds": 60,
    "max_lag_seconds": 300,
    "max_gap_seconds": 600,
}


def test_endpoint_detects_positive_lag_and_direction(client):
    _ingest(client, "CASE-S", _wave_readings("CASE-S", 0))
    _ingest(client, "CASE-T", _wave_readings("CASE-T", 2))
    r = client.get(
        "/api/v1/propagation",
        params={**BASE_PARAMS, "source_sn": "CASE-S", "target_sn": "CASE-T",
                "dimension": "temperature"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source_sn"] == "CASE-S"
    assert body["target_sn"] == "CASE-T"
    assert body["dimension"] == "temperature"
    assert body["best_lag_seconds"] == 120.0
    assert body["best_correlation"] == pytest.approx(1.0, abs=1e-9)
    assert body["direction"] == "source_to_target"
    assert body["grid_point_count"] == 10
    lags = body["lags"]
    assert [l["lag_seconds"] for l in lags] == sorted(l["lag_seconds"] for l in lags)
    best = next(l for l in lags if l["lag_seconds"] == 120.0)
    assert best["available"] is True
    # 不存在 NaN（JSON 中 NaN 不合法；响应可正常解析即满足）
    assert all(l["correlation"] is None or isinstance(l["correlation"], (int, float)) for l in lags)


def test_endpoint_negative_lag(client):
    _ingest(client, "CASE-S2", _wave_readings("CASE-S2", 2))
    _ingest(client, "CASE-T2", _wave_readings("CASE-T2", 0))
    r = client.get(
        "/api/v1/propagation",
        params={**BASE_PARAMS, "source_sn": "CASE-S2", "target_sn": "CASE-T2",
                "dimension": "temperature"},
    )
    body = r.json()
    assert body["best_lag_seconds"] == -120.0
    assert body["direction"] == "target_to_source"


def test_endpoint_no_overlap_is_unavailable_200(client):
    _ingest(client, "CASE-A", [
        _reading("CASE-A", "2026-09-01T00:00:00Z", 20.0, 50.0),
        _reading("CASE-A", "2026-09-01T00:10:00Z", 24.0, 50.0),
    ])
    _ingest(client, "CASE-B", [
        _reading("CASE-B", "2026-09-02T00:00:00Z", 20.0, 50.0),
        _reading("CASE-B", "2026-09-02T00:10:00Z", 24.0, 50.0),
    ])
    r = client.get(
        "/api/v1/propagation",
        params={
            "source_sn": "CASE-A", "target_sn": "CASE-B", "dimension": "temperature",
            "start": "2026-09-01T00:00:00Z", "end": "2026-09-03T00:00:00Z",
            "step_seconds": 60, "max_lag_seconds": 300, "max_gap_seconds": 600,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["direction"] == "unavailable"
    assert body["best_lag_seconds"] is None
    assert body["best_correlation"] is None
    assert body["overlap_start"] is None
    assert body["overlap_end"] is None
    assert body["grid_point_count"] == 0
    assert all(l["available"] is False and l["correlation"] is None for l in body["lags"])


def test_endpoint_unknown_device_404(client):
    _ingest(client, "CASE-KNOWN", _wave_readings("CASE-KNOWN", 0))
    r = client.get(
        "/api/v1/propagation",
        params={**BASE_PARAMS, "source_sn": "CASE-KNOWN", "target_sn": "CASE-GHOST",
                "dimension": "temperature"},
    )
    assert r.status_code == 404
    r = client.get(
        "/api/v1/propagation",
        params={**BASE_PARAMS, "source_sn": "CASE-GHOST", "target_sn": "CASE-KNOWN",
                "dimension": "temperature"},
    )
    assert r.status_code == 404


def test_endpoint_validation_errors(client):
    _ingest(client, "CASE-V1", _wave_readings("CASE-V1", 0))
    _ingest(client, "CASE-V2", _wave_readings("CASE-V2", 0))
    base = {
        "source_sn": "CASE-V1", "target_sn": "CASE-V2", "dimension": "temperature",
        **BASE_PARAMS,
    }

    def status(**over):
        p = {**base, **over}
        return client.get("/api/v1/propagation", params=p).status_code

    # source 与 target 相同
    assert status(source_sn="CASE-V1", target_sn="CASE-V1") == 422
    # start 晚于 end
    assert status(start="2026-09-02T00:00:00Z", end="2026-09-01T00:00:00Z") == 422
    # step 非正
    assert status(step_seconds=0) == 422
    assert status(step_seconds=-10) == 422
    # max_lag 负
    assert status(max_lag_seconds=-1) == 422
    # max_gap 非正
    assert status(max_gap_seconds=0) == 422
    # 非法维度
    assert status(dimension="pressure") == 422
    # 缺少必填参数
    assert client.get("/api/v1/propagation", params={"source_sn": "CASE-V1"}).status_code == 422


def test_endpoint_max_gap_controls_invalidation(client):
    # target 在中间断档 20 分钟；max_gap=600 时断档内点无效
    rows = [
        _reading("CASE-G1", "2026-09-01T00:%02d:00Z" % i, WAVE[i], 50.0)
        for i in range(12)
    ]
    _ingest(client, "CASE-G1", rows)
    target_rows = [
        _reading("CASE-G2", "2026-09-01T00:00:00Z", 20.0, 50.0),
        _reading("CASE-G2", "2026-09-01T00:01:00Z", 22.0, 50.0),
        _reading("CASE-G2", "2026-09-01T00:21:00Z", 21.0, 50.0),
        _reading("CASE-G2", "2026-09-01T00:22:00Z", 24.0, 50.0),
    ]
    _ingest(client, "CASE-G2", target_rows)
    params = {
        "source_sn": "CASE-G1", "target_sn": "CASE-G2", "dimension": "temperature",
        "start": "2026-09-01T00:00:00Z", "end": "2026-09-01T00:11:00Z",
        "step_seconds": 60, "max_lag_seconds": 0, "max_gap_seconds": 600,
    }
    body = client.get("/api/v1/propagation", params=params).json()
    # target 有效点：0,1 分钟（11 分钟超出 target 覆盖前的断档，落在 1→21 间隔内）
    assert body["target_valid_count"] == 2

    # 放宽 max_gap 到 2000 秒：断档内点恢复有效
    params["max_gap_seconds"] = 2000
    body2 = client.get("/api/v1/propagation", params=params).json()
    assert body2["target_valid_count"] == body2["grid_point_count"]


def test_endpoint_deterministic(client):
    _ingest(client, "CASE-D1", _wave_readings("CASE-D1", 0))
    _ingest(client, "CASE-D2", _wave_readings("CASE-D2", 2))
    params = {**BASE_PARAMS, "source_sn": "CASE-D1", "target_sn": "CASE-D2",
              "dimension": "temperature"}
    first = client.get("/api/v1/propagation", params=params).json()
    second = client.get("/api/v1/propagation", params=params).json()
    assert first == second


# ---------- 接口集成：微秒精度、非有限参数与规模上限 ----------


def _iso_offset(seconds: float) -> str:
    return (BASE + dt.timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _ingest_endpoint_devices(client, prefix: str, span_seconds: float):
    """两柜仅在窗口两端上报读数（极大 max_gap 时覆盖整窗）。"""
    for sn, base_t in ((f"{prefix}-S", 20.0), (f"{prefix}-T", 21.0)):
        _ingest(client, sn, [
            _reading(sn, _iso_offset(0), base_t, 50.0),
            _reading(sn, _iso_offset(span_seconds), base_t + 5.0, 50.0),
        ])


def test_endpoint_sub_microsecond_step_422(client):
    _ingest(client, "CASE-US1", _wave_readings("CASE-US1", 0))
    _ingest(client, "CASE-US2", _wave_readings("CASE-US2", 0))
    p = {
        "source_sn": "CASE-US1", "target_sn": "CASE-US2", "dimension": "temperature",
        "start": _iso_offset(0), "end": _iso_offset(600),
        "max_lag_seconds": 0, "max_gap_seconds": 600,
    }
    # 小于半微秒的正步长取整为零（原先会除零）
    r = client.get("/api/v1/propagation", params={**p, "step_seconds": 1e-7})
    assert r.status_code == 422
    assert "微秒" in r.json()["detail"]["message"]
    # 0.5µs 按偶数取整为 0
    r = client.get("/api/v1/propagation", params={**p, "step_seconds": 0.5e-6})
    assert r.status_code == 422


def test_endpoint_nonfinite_params_422(client):
    _ingest(client, "CASE-NF1", _wave_readings("CASE-NF1", 0))
    _ingest(client, "CASE-NF2", _wave_readings("CASE-NF2", 0))
    p = {
        "source_sn": "CASE-NF1", "target_sn": "CASE-NF2", "dimension": "temperature",
        "start": _iso_offset(0), "end": _iso_offset(600),
        "step_seconds": 60, "max_lag_seconds": 300, "max_gap_seconds": 600,
    }
    for name, value in [
        ("step_seconds", "inf"),
        ("step_seconds", "1e999"),
        ("step_seconds", "nan"),
        ("max_lag_seconds", "inf"),
        ("max_lag_seconds", "nan"),
        ("max_gap_seconds", "inf"),
        ("max_gap_seconds", "nan"),
    ]:
        r = client.get("/api/v1/propagation", params={**p, name: value})
        assert r.status_code == 422, (name, value, r.text)


def test_endpoint_effective_step_echoed(client):
    _ingest(client, "CASE-ES1", _wave_readings("CASE-ES1", 0))
    _ingest(client, "CASE-ES2", _wave_readings("CASE-ES2", 0))
    r = client.get(
        "/api/v1/propagation",
        params={
            "source_sn": "CASE-ES1", "target_sn": "CASE-ES2",
            "dimension": "temperature",
            "start": _iso_offset(0), "end": _iso_offset(660),
            # 60s 加 0.4 微秒 → 微秒取整后生效步长仍为 60s
            "step_seconds": 60.0000004, "max_lag_seconds": 300,
            "max_gap_seconds": 600,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["step_seconds"] == 60.0
    assert [lag["lag_seconds"] for lag in body["lags"]] == [
        -300.0, -240.0, -180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0, 240.0, 300.0
    ]


def test_endpoint_grid_limit_boundary_and_overflow(client):
    # 1s 步长：跨度 9999s → 10000 点放行；10000s → 10001 点 422
    _ingest_endpoint_devices(client, "GRID-OK", 9999)
    r = client.get("/api/v1/propagation", params={
        "source_sn": "GRID-OK-S", "target_sn": "GRID-OK-T", "dimension": "temperature",
        "start": _iso_offset(0), "end": _iso_offset(9999),
        "step_seconds": 1, "max_lag_seconds": 0, "max_gap_seconds": 1e12,
    })
    assert r.status_code == 200, r.text
    assert r.json()["grid_point_count"] == 10000

    _ingest_endpoint_devices(client, "GRID-BAD", 10000)
    r = client.get("/api/v1/propagation", params={
        "source_sn": "GRID-BAD-S", "target_sn": "GRID-BAD-T", "dimension": "temperature",
        "start": _iso_offset(0), "end": _iso_offset(10000),
        "step_seconds": 1, "max_lag_seconds": 0, "max_gap_seconds": 1e12,
    })
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["grid_point_count"] == 10001
    assert detail["max_grid_points"] == 10000


def test_endpoint_lag_row_limit_boundary_and_overflow(client):
    _ingest(client, "CASE-LR1", _wave_readings("CASE-LR1", 0))
    _ingest(client, "CASE-LR2", _wave_readings("CASE-LR2", 0))
    p = {
        "source_sn": "CASE-LR1", "target_sn": "CASE-LR2", "dimension": "temperature",
        "start": _iso_offset(0), "end": _iso_offset(660),
        "step_seconds": 60, "max_gap_seconds": 600,
    }
    # 边界：max_lag=100.9 步 → kmax=100 → 201 行
    r = client.get("/api/v1/propagation", params={**p, "max_lag_seconds": 6054})
    assert r.status_code == 200, r.text
    assert len(r.json()["lags"]) == 201
    # 超限：kmax=101 → 203 行
    r = client.get("/api/v1/propagation", params={**p, "max_lag_seconds": 6060})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["lag_row_count"] == 203
    assert detail["max_lag_rows"] == 201


def test_endpoint_pair_work_boundary_and_overflow(client):
    # 1s 步长、kmax=100：n=5025 → 工作量 999_925（放行）；n=5026 → 1_000_126（拒绝）
    _ingest_endpoint_devices(client, "WORK-OK", 5024)
    r = client.get("/api/v1/propagation", params={
        "source_sn": "WORK-OK-S", "target_sn": "WORK-OK-T", "dimension": "temperature",
        "start": _iso_offset(0), "end": _iso_offset(5024),
        "step_seconds": 1, "max_lag_seconds": 100, "max_gap_seconds": 1e12,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grid_point_count"] == 5025
    assert len(body["lags"]) == 201

    _ingest_endpoint_devices(client, "WORK-BAD", 5025)
    r = client.get("/api/v1/propagation", params={
        "source_sn": "WORK-BAD-S", "target_sn": "WORK-BAD-T", "dimension": "temperature",
        "start": _iso_offset(0), "end": _iso_offset(5025),
        "step_seconds": 1, "max_lag_seconds": 100, "max_gap_seconds": 1e12,
    })
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["pair_work"] == 1_000_126
    assert detail["max_pair_work"] == 1_000_000


def test_endpoint_lag_rows_checked_before_overlap(client):
    # 两柜读数完全不重叠，但时滞行数无界仍应在分配前 422
    _ingest(client, "CASE-NOLAG1", [
        _reading("CASE-NOLAG1", _iso_offset(0), 20.0, 50.0),
        _reading("CASE-NOLAG1", _iso_offset(60), 24.0, 50.0),
    ])
    _ingest(client, "CASE-NOLAG2", [
        _reading("CASE-NOLAG2", _iso_offset(3600), 20.0, 50.0),
        _reading("CASE-NOLAG2", _iso_offset(3660), 24.0, 50.0),
    ])
    r = client.get("/api/v1/propagation", params={
        "source_sn": "CASE-NOLAG1", "target_sn": "CASE-NOLAG2",
        "dimension": "temperature",
        "start": _iso_offset(0), "end": _iso_offset(4000),
        "step_seconds": 60, "max_lag_seconds": 10000, "max_gap_seconds": 600,
    })
    assert r.status_code == 422
    assert r.json()["detail"]["lag_row_count"] > 201
