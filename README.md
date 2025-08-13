# Install requirements
Create a virtual environment
- With conda:
```bash
conda create -n aoi_venv python=3.10
conda activate aoi_venv
```
- With venv:
```bash
python3.10 -m venv aoi_venv
source aoi_venv/bin/activate
```

Manually install torch first
- For CPU only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

- For GPU support (CUDA 12.3):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu123
```

Install the remaining dependencies
```bash
pip install -e .
```

