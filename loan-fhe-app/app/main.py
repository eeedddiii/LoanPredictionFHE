import time

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .schemas import LoanInput
from .fhe_logic import predict_plain, predict_fhe_ckks

# FastAPI 앱 인스턴스
app = FastAPI(
    title="Loan FHE Demo",
    description="Plain vs FHE Loan Default Prediction",
    version="1.0.0",
)

# 템플릿 디렉토리 설정 (Docker 컨테이너 기준 /app/templates)
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    메인 페이지: 대출 정보를 입력하고
    평문 / FHE 예측 결과를 비교할 수 있는 웹 UI.
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
        },
    )


@app.post("/web_predict", response_class=HTMLResponse)
async def web_predict(
    request: Request,
    Age: float = Form(...),
    Income: float = Form(...),
    LoanAmount: float = Form(...),
    CreditScore: float = Form(...),
    MonthsEmployed: float = Form(...),
    NumCreditLines: float = Form(...),
    InterestRate: float = Form(...),
    LoanTerm: float = Form(...),
    DTIRatio: float = Form(...),
    Education: float = Form(...),
    EmploymentType: float = Form(...),
    MaritalStatus: float = Form(...),
    HasMortgage: float = Form(...),
    HasDependents: float = Form(...),
    LoanPurpose: float = Form(...),
    HasCoSigner: float = Form(...),
):
    """
    웹 폼에서 넘어온 데이터를 이용해
    - 평문 Logistic 모델
    - FHE(CKKS) Logistic 모델
    두 가지로 동시에 추론하고 결과 + 추론 시간을 화면에 렌더링.
    """
    # 1) Form 데이터 → Pydantic 모델로 변환
    data = LoanInput(
        Age=Age,
        Income=Income,
        LoanAmount=LoanAmount,
        CreditScore=CreditScore,
        MonthsEmployed=MonthsEmployed,
        NumCreditLines=NumCreditLines,
        InterestRate=InterestRate,
        LoanTerm=LoanTerm,
        DTIRatio=DTIRatio,
        Education=Education,
        EmploymentType=EmploymentType,
        MaritalStatus=MaritalStatus,
        HasMortgage=HasMortgage,
        HasDependents=HasDependents,
        LoanPurpose=LoanPurpose,
        HasCoSigner=HasCoSigner,
    )

    # 2) 평문 모델 추론 + 시간 측정
    start_plain = time.time()
    plain_prob, plain_label = predict_plain(data)
    plain_time = time.time() - start_plain

    # 3) FHE(CKKS) 모델 추론 + 시간 측정
    start_fhe = time.time()
    fhe_prob, fhe_label = predict_fhe_ckks(data)
    fhe_time = time.time() - start_fhe

    # 4) 템플릿에 넘길 결과 구성
    result = {
        "inputs": data.dict(),
        "plain_prob": plain_prob,
        "plain_label": plain_label,
        "plain_time": plain_time,
        "fhe_prob": fhe_prob,
        "fhe_label": fhe_label,
        "fhe_time": fhe_time,
    }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
        },
    )


# ============================
# JSON API 버전 (실험/테스트용)
# ============================

@app.post("/predict_plain")
async def api_predict_plain(data: LoanInput):
    """
    JSON 입력(LoanInput)을 받아 평문 모델로만 추론하는 엔드포인트.
    논문 실험 자동화, 성능 측정 등에 활용 가능.
    """
    start = time.time()
    prob, label = predict_plain(data)
    elapsed = time.time() - start

    return {
        "mode": "plain",
        "probability_default": prob,
        "label": label,
        "inference_time_sec": elapsed,
    }


@app.post("/predict_fhe_ckks")
async def api_predict_fhe(data: LoanInput):
    """
    JSON 입력(LoanInput)을 받아 FHE(CKKS) 모델로만 추론하는 엔드포인트.
    """
    start = time.time()
    prob, label = predict_fhe_ckks(data)
    elapsed = time.time() - start

    return {
        "mode": "fhe_ckks",
        "probability_default": prob,
        "label": label,
        "inference_time_sec": elapsed,
    }