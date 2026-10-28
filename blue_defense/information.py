from dataclasses import replace
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .base import BlueObservation, DefendedAsset, InterceptorState, ThreatTrack, Vector3, distance_m


class BlueInformationStore:
    TRACK_LIFETIME_MS = 30_000

    def __init__(self):
        self._tracks: dict[int, ThreatTrack] = {}
        self._interceptors: tuple[InterceptorState, ...] = ()
        self._assets: tuple[DefendedAsset, ...] = ()
        self._launched: dict[int, tuple[int, ...]] = {}

    def reset(self) -> None:
        self._tracks.clear()
        self._interceptors = ()
        self._assets = ()
        self._launched.clear()

    def receive_tracks(self, tracks: Iterable[ThreatTrack], sim_time: float) -> None:
        received = tuple(tracks)
        for track in received:
            if (track.target_id <= 0 or track.source_id <= 0
                    or track.type not in (21000, 21001, 21002)
                    or not math.isfinite(track.detect_time)
                    or track.detect_time < 0 or track.detect_time > sim_time):
                raise ValueError("蓝方探测记录的身份、来源或时间无效")
            if not all(math.isfinite(value) for vector in (track.lla, track.pos_ecf, track.vel_ecf)
                       for value in (vector.x, vector.y, vector.z)):
                raise ValueError("蓝方探测记录包含无效坐标或速度")
        self._expire(sim_time)
        for track in received:
            if sim_time - track.detect_time > self.TRACK_LIFETIME_MS:
                continue
            previous = self._tracks.get(track.target_id)
            if previous is not None and (track.detect_time, -track.source_id) <= (
                    previous.detect_time, -previous.source_id):
                continue
            self._tracks[track.target_id] = track

    def update_own_state(self, interceptors: Iterable[InterceptorState],
                         assets: Iterable[DefendedAsset],
                         launched_map: Mapping[int, Sequence[int]]) -> None:
        self._interceptors = tuple(interceptors)
        self._assets = tuple(assets)
        self._launched = {target_id: tuple(ids) for target_id, ids in launched_map.items()}

    def get_track(self, target_id: int, sim_time: float) -> ThreatTrack | None:
        track = self._tracks.get(target_id)
        if track is None:
            return None
        if sim_time - track.detect_time > self.TRACK_LIFETIME_MS:
            del self._tracks[target_id]
            return None
        return self._predict(track, sim_time)

    def snapshot(self, sim_time: float) -> BlueObservation:
        self._expire(sim_time)
        return BlueObservation(
            sim_time=sim_time,
            targets=tuple(self._predict(self._tracks[target_id], sim_time)
                          for target_id in sorted(self._tracks)),
            interceptors=self._interceptors,
            assets=self._assets,
            launched_map=MappingProxyType(dict(self._launched)),
        )

    def nearest_track(self, position: Vector3, max_range_m: float,
                      sim_time: float) -> ThreatTrack | None:
        targets = self.snapshot(sim_time).targets
        candidates = [(distance_m(position, track.pos_ecf), track.target_id, track)
                      for track in targets]
        candidates = [item for item in candidates if item[0] <= max_range_m]
        return min(candidates, key=lambda item: item[:2])[2] if candidates else None

    def _expire(self, sim_time: float) -> None:
        self._tracks = {target_id: track for target_id, track in self._tracks.items()
                        if sim_time - track.detect_time <= self.TRACK_LIFETIME_MS}

    @staticmethod
    def _predict(track: ThreatTrack, sim_time: float) -> ThreatTrack:
        # 时间使用毫秒，位置预测使用秒；探测时间和最后观测速度保持原值。
        elapsed_seconds = max(0.0, sim_time - track.detect_time) / 1000.0
        return replace(track, pos_ecf=Vector3(
            track.pos_ecf.x + track.vel_ecf.x * elapsed_seconds,
            track.pos_ecf.y + track.vel_ecf.y * elapsed_seconds,
            track.pos_ecf.z + track.vel_ecf.z * elapsed_seconds,
        ))
