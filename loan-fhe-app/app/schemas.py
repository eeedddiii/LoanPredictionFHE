from pydantic import BaseModel

class LoanInput(BaseModel):
    Age: float
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: float
    NumCreditLines: float
    InterestRate: float
    LoanTerm: float
    DTIRatio: float
    Education: float
    EmploymentType: float
    MaritalStatus: float
    HasMortgage: float
    HasDependents: float
    LoanPurpose: float
    HasCoSigner: float