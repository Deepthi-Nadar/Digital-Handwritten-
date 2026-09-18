
import os
import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from fastapi.templating import Jinja2Templates

from locker_system import verify_login




BASE_DIR = Path(__file__).resolve().parent

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_FOLDER = STATIC_DIR / "uploads"
MODEL_PATH = BASE_DIR / "digit_model.h5"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)



app = FastAPI(title="Handwritten Digit Recognition")



app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "digitsecret")
)


templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)




model = tf.keras.models.load_model(str(MODEL_PATH))



@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={}
    )



@app.post("/", response_class=HTMLResponse)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):

    if verify_login(username, password):

        request.session["user"] = username

        return RedirectResponse(
            url="/predict",
            status_code=303
        )

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "error": "Invalid credentials"
        }
    )


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):

    if "user" not in request.session:

        return RedirectResponse(
            url="/",
            status_code=303
        )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )



@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    image: UploadFile = File(...)
):

    if "user" not in request.session:

        return RedirectResponse(
            url="/",
            status_code=303
        )

    if not image.filename:

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": "Please select an image."
            }
        )

    extension = Path(image.filename).suffix.lower()

    if extension not in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": "Please upload a valid image file."
            }
        )

    filename = f"{uuid.uuid4().hex}{extension}"

    file_path = UPLOAD_FOLDER / filename

    # Save uploaded image
    contents = await image.read()

    with open(file_path, "wb") as f:
        f.write(contents)

    try:

        img = Image.open(file_path).convert("L")

        img = img.resize((28, 28))

        img = np.array(img, dtype=np.float32)

        img = img / 255.0

        img = img.reshape(1, 28, 28)


        prediction = model.predict(img, verbose=0)

        digit = int(np.argmax(prediction))

        # URL accessible from the browser
        image_url = f"/static/uploads/{filename}"

        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "digit": digit,
                "image": image_url
            }
        )

    except Exception:

        # Remove invalid or unreadable uploaded files
        if file_path.exists():
            file_path.unlink()

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": "Unable to process this image. Please try another one."
            }
        )


@app.get("/logout")
async def logout(request: Request):

    request.session.pop("user", None)

    return RedirectResponse(
        url="/",
        status_code=303
    )
