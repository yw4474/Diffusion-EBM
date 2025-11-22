# Applied Generative AI Project – Helper Library + FastAPI + Docker

This repository contains a reusable PyTorch helper library (`helper_lib`) and a FastAPI API server (under `app/`) that expose several generative models:

- CNN classifier (specCNN for CIFAR-10)
- GAN image generator (MNIST)
- Diffusion image generator (MNIST)
- Energy-Based Model (EBM) image generator (MNIST)

The project is designed to satisfy the course requirements:
- The API can be successfully queried to run the model implemented for the assignment.
- The submission includes a Docker deployment that runs a FastAPI server with the added API endpoints on the instructor’s machine.

---

## 1. Project Structure

```text
.
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── router.py
│   └── services/
│       ├── __init__.py
│       ├── text_gen_service.py        # simple text generator (Module 3)
│       ├── image_classifier.py        # CNN classifier
│       ├── gan_service.py             # GAN generator service
│       ├── diffusion_service.py       # Diffusion generator service
│       └── ebm_service.py             # EBM generator service
├── helper_lib/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── generator.py
│   ├── model.py
│   ├── trainer.py
│   └── utils.py
├── scripts/
│   ├── train_gan_mnist.py
│   └── test_ebm.py
├── weights/                           # model weights (if available)
│   ├── gan_mnist_gen.pt              # GAN generator (MNIST)
│   ├── speccnn_cifar10.pt            # CNN classifier (CIFAR-10, may be partially trained)
│   ├── diffusion_mnist.pt            # (optional) trained diffusion model
│   └── ebm_mnist.pt                  # (optional) trained EBM model
├── requirements.txt
├── Dockerfile
└── README.md
