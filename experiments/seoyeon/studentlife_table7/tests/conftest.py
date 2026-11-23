"""테스트가 어디서 실행되든 프로젝트 폴더의 모듈을 찾도록 경로를 더한다."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
