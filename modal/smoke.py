import modal
app = modal.App("phasesplit-smoke")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.6.0", "numpy", "scikit-learn", "scipy"))

@app.function(image=img, gpu="A10G", timeout=600)
def check():
    import torch, os
    return dict(torch=torch.__version__, cuda=torch.cuda.is_available(),
                name=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                cpus=os.cpu_count())

@app.local_entrypoint()
def main():
    print("GPU check:", check.remote())
