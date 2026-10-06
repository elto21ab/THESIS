from .base import Chunk, ChunkCfg
from . import day_heuristic, fixed

CHUNKERS = {
    "fixed": fixed.chunk_thread,
    "day": day_heuristic.chunk_thread,
}
