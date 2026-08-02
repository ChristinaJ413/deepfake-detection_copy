from fastapi import FastAPI, UploadFile, File, Request
from inference import predict_deepfake
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

def get_real_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

app = FastAPI()

limiter = Limiter(key_func=get_real_client_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# implementing cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://deepfake-detection-frontend.s3-website-us-east-1.amazonaws.com",
    "https://dbnb07cpxph7w.cloudfront.net",
    ],
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_deepfake(image_bytes)
    return result