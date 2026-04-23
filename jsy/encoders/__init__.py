import sys
import os

# 1. 자기 자신(encoders 폴더)을 sys.path에 등록
current_package_path = os.path.dirname(os.path.abspath(__file__))
if current_package_path not in sys.path:
    sys.path.append(current_package_path)