# app/api/router.py
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import zipfile, time

from app.services.gan_service import GANSampler
from app.services.diffusion_service import DiffusionSampler
from app.services.ebm_service import EBMSampler

router = APIRouter()

gan = GANSampler(weights_path="weights/gan_mnist_gen.pt", device="cpu")
diffusion = DiffusionSampler(weights_path="weights/diffusion_mnist.pt", device="cpu")
ebm = EBMSampler(weights_path="weights/ebm_mnist.pt", device="cpu")

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/gan/generate")
def gan_generate(n: int = 4):
    try:
        n = max(1, min(int(n), 16))  # 安全上限
        images = gan.sample_base64_pngs(n)
        return {"count": len(images), "images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/gan/generate-files")
def gan_generate_files(
    n: int = Query(4, ge=1, le=64),
    out_dir: str = Query("weights/gan_samples")
):
    try:
        paths = gan.sample_png_files(n=n, out_dir=out_dir, prefix="gan")
        return JSONResponse({"count": len(paths), "dir": str(Path(out_dir).resolve()), "files": paths})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/gan/generate-zip")
def gan_generate_zip(
    n: int = Query(8, ge=1, le=64),
    out_dir: str = Query("weights/gan_samples"),
    zip_dir: str = Query("weights/gan_exports")
):
    try:
        paths = gan.sample_png_files(n=n, out_dir=out_dir, prefix="gan")
        Path(zip_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        zip_path = Path(zip_dir) / f"gan_samples_{ts}.zip"

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in paths:
                zf.write(p, arcname=Path(p).name)

        return FileResponse(
            path=str(zip_path.resolve()),
            media_type="application/zip",
            filename=zip_path.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/diffusion/generate")
def diffusion_generate(
    n: int = Query(4, ge=1, le=32),
    steps: int = Query(200, ge=10, le=1000),
    seed: int | None = None,
):
    try:
        images = diffusion.sample_base64_pngs(n=n, steps=steps, seed=seed)
        return {
            "count": len(images),
            "images": images,
            "steps": steps,
            "seed": seed,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/diffusion/generate-files")
def diffusion_generate_files(
    n: int = Query(8, ge=1, le=64),
    steps: int = Query(200, ge=10, le=1000),
    out_dir: str = Query("weights/diffusion_samples"),
    prefix: str = Query("diff"),
    seed: int | None = None,
):
    try:
        paths = diffusion.save_pngs(
            n=n, steps=steps, out_dir=out_dir, prefix=prefix, seed=seed
        )
        return {
            "count": len(paths),
            "dir": str(Path(out_dir).resolve()),
            "files": paths,
            "steps": steps,
            "seed": seed,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ebm/generate")
def ebm_generate(
    n: int = Query(4, ge=1, le=32),
    steps: int = Query(60, ge=10, le=200),
    step_size: float = Query(0.1, gt=0.0),
    noise_scale: float = Query(0.01, ge=0.0),
    seed: int | None = None,
):
    try:
        images = ebm.sample_base64_pngs(
            n=n,
            steps=steps,
            step_size=step_size,
            noise_scale=noise_scale,
            seed=seed,
        )
        return {
            "count": len(images),
            "images": images,
            "steps": steps,
            "step_size": step_size,
            "noise_scale": noise_scale,
            "seed": seed,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ebm/generate-files")
def ebm_generate_files(
    n: int = Query(8, ge=1, le=64),
    steps: int = Query(60, ge=10, le=200),
    step_size: float = Query(0.1, gt=0.0),
    noise_scale: float = Query(0.01, ge=0.0),
    out_dir: str = Query("weights/ebm_samples"),
    prefix: str = Query("ebm"),
    seed: int | None = None,
):
    try:
        paths = ebm.save_pngs(
            n=n,
            steps=steps,
            step_size=step_size,
            noise_scale=noise_scale,
            seed=seed,
            out_dir=out_dir,
            prefix=prefix,
        )
        return {
            "count": len(paths),
            "dir": str(Path(out_dir).resolve()),
            "files": paths,
            "steps": steps,
            "step_size": step_size,
            "noise_scale": noise_scale,
            "seed": seed,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
