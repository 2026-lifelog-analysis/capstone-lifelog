"""pytest가 레포 맨 위 폴더를 기준으로 src를 찾도록 경로를 추가함."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
