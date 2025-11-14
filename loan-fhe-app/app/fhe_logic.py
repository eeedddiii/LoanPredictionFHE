# app/fhe_logic.py

"""
평문 Logistic Regression + FHE(CKKS) 기반 Logistic 추론 모듈.

- Colab에서 학습한 Logistic Regression + StandardScaler 파라미터를
  logistic_fhe_params.pkl에서 로드해서 사용한다.
- 평문 추론: numpy.dot + sigmoid
- FHE 추론: Pyfhel CKKS를 이용해 x_i 암호화, w_i/b plaintext encode 후
  ct_x * pt_w, rescale_to_next, sum, decrypt, sigmoid 수행.
"""

import os
from typing import Tuple

import joblib
import numpy as np

from Pyfhel import Pyfhel

from .schemas import LoanInput

# ==========================
# 전역 변수 (한 번만 초기화)
# ==========================

FEATURE_NAMES = None
SCALER_MEAN = None
SCALER_SCALE = None
W = None
B = None
HE: Pyfhel | None = None  # CKKS 컨텍스트/키


# ==========================
# 내부 유틸
# ==========================

def _load_params_once():
    """
    - logistic_fhe_params.pkl 에서
      feature_names, scaler_mean, scaler_scale, coef(w), intercept(b) 로드
    - Pyfhel CKKS 컨텍스트 및 키 생성
    """
    global FEATURE_NAMES, SCALER_MEAN, SCALER_SCALE, W, B, HE

    if FEATURE_NAMES is not None and HE is not None:
        # 이미 초기화된 상태
        return

    # 1) 모델 파라미터 로드
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),  # /app/app → /app
        "model",
        "logistic_fhe_params.pkl"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model params file not found: {model_path}")

    params = joblib.load(model_path)

    FEATURE_NAMES = params["feature_names"]
    SCALER_MEAN = np.array(params["scaler_mean"], dtype=float)
    SCALER_SCALE = np.array(params["scaler_scale"], dtype=float)
    W = np.array(params["coef"], dtype=float)
    B = float(params["intercept"])

    # 2) Pyfhel CKKS 컨텍스트 초기화
    he = Pyfhel()
    he.contextGen(
        scheme="CKKS",
        n=2**14,
        scale=2**30,
        qi_sizes=[60, 30, 30, 30, 60],
    )
    he.keyGen()

    HE = he


def _loaninput_to_vector(data: LoanInput) -> np.ndarray:
    """
    LoanInput → feature_names 순서대로 numpy 벡터로 변환
    """
    if FEATURE_NAMES is None:
        raise RuntimeError("FEATURE_NAMES not initialized. Call _load_params_once() first.")

    values = [getattr(data, name) for name in FEATURE_NAMES]
    return np.array(values, dtype=float)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# ==========================
# 평문 Logistic 추론
# ==========================

def predict_plain(data: LoanInput) -> Tuple[float, int]:
    """
    평문 Logistic Regression 추론.
    - 입력: LoanInput
    - 출력: (probability_default, label)
    """
    _load_params_once()

    x_raw = _loaninput_to_vector(data)

    x_scaled = (x_raw - SCALER_MEAN) / SCALER_SCALE

    score = float(np.dot(W, x_scaled) + B)
    prob = _sigmoid(score)
    label = 1 if prob >= 0.5 else 0

    return prob, label


# ==========================
# FHE (CKKS) Logistic 추론 (하드코어 버전)
# ==========================

def _fhe_logistic_predict_one_ckks(x_raw: np.ndarray) -> Tuple[float, int]:
    """
    CKKS 기반 Logistic Regression 추론 (단일 샘플).

    - x_raw: 원본 feature 벡터 (1D numpy array, scaling 전)
    - 절차:
        1) x_scaled = (x_raw - mean) / scale
        2) 각 x_i를 encryptFrac → ciphertext ct_x_i
        3) 각 w_i를 encodeFrac → plaintext pt_w_i
        4) term_i = ct_x_i * pt_w_i (ciphertext-plaintext 곱)
        5) rescale_to_next(term_i) 로 scale/level 조정
        6) score_ct = Σ term_i
        7) bias b를 plaintext pt_b 로 encode 후 score_ct에 더함
        8) decryptFrac(score_ct) → score → sigmoid → 확률/라벨
    """
    if HE is None:
        raise RuntimeError("HE (Pyfhel) is not initialized. Call _load_params_once() first.")

    he: Pyfhel = HE

    # 1) numpy 배열 변환 + scaling
    x_raw = np.array(x_raw, dtype=float)
    x_scaled = (x_raw - SCALER_MEAN) / SCALER_SCALE

    # 2) bias b 를 CKKS plaintext 로 encode
    pt_b = he.encodeFrac(
        np.array([float(B)], dtype=np.float64),
        scale=2**30
    )

    score_ct = None

    # 3~6) 각 특성에 대해 term_i 계산 및 누적
    for xi, wi in zip(x_scaled, W):
        # 3-1) x_i 암호화
        ct_xi = he.encryptFrac(
            np.array([float(xi)], dtype=np.float64),
            scale=2**30
        )

        # 3-2) w_i plaintext encode
        pt_wi = he.encodeFrac(
            np.array([float(wi)], dtype=np.float64),
            scale=2**30
        )

        # 4) term_i = ct_xi * pt_wi
        term_ct = ct_xi * pt_wi

        # 5) rescale_to_next (in-place)
        he.rescale_to_next(term_ct)

        # 6) 누적합
        if score_ct is None:
            score_ct = term_ct
        else:
            score_ct += term_ct

    # 7) bias 더하기
    score_ct += pt_b

    # 8) 복호화 후 sigmoid
    score_vec = he.decryptFrac(score_ct)
    score = float(score_vec[0])

    prob = _sigmoid(score)
    label = 1 if prob >= 0.5 else 0

    return prob, label


def predict_fhe_ckks(data: LoanInput) -> Tuple[float, int]:
    """
    FHE(CKKS) 기반 Logistic Regression 추론 (웹/API에서 호출되는 외부용 함수).
    """
    _load_params_once()

    x_raw = _loaninput_to_vector(data)
    prob, label = _fhe_logistic_predict_one_ckks(x_raw)
    return prob, label