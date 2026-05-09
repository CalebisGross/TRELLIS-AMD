import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

# AMD/ROCm: enable AOTriton experimental attention paths used by PyTorch SDPA.
# Must be set before `import torch`. Harmless on CUDA builds (only the ROCm
# path reads the flag).
os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '1')

import torch

# Configure torchsparse for HIP/ROCm compatibility (mirrors app.py).
# Use GatherScatter dataflow instead of ImplicitGEMM (PTX-only).
if os.environ.get('SPARSE_BACKEND') == 'torchsparse' and hasattr(torch.version, 'hip'):
    try:
        from torchsparse.nn.functional.conv.conv_config import (
            Dataflow, set_global_conv_config, _default_conv_config
        )
        from torchsparse.nn.functional.conv.conv_mode import set_conv_mode
        hip_config = _default_conv_config.copy()
        hip_config['dataflow'] = Dataflow.GatherScatter
        set_global_conv_config(hip_config)
        set_conv_mode(0)
        print("[TORCHSPARSE] Configured for HIP: GatherScatter dataflow, mode0")
    except Exception as e:
        print(f"[TORCHSPARSE] Warning: Could not configure for HIP: {e}")

import imageio
import gc
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


def _gpu_mem(label):
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"[mem {label}] free={free_b/2**30:.2f} GiB used={(total_b-free_b)/2**30:.2f} GiB")


def _move_unused_models_to_cpu(pipe, keep):
    """Move every pipeline submodel to CPU except those in `keep`.
    Frees GPU memory for the remaining decode step. Required on 16 GB cards
    where the full pipeline footprint exceeds VRAM.
    """
    for name, m in list(pipe.models.items()):
        if name not in keep and hasattr(m, 'cpu'):
            m.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()
# Pin pipeline.device to cuda regardless of which submodels we offload
# to CPU between phases. Without this, pipeline.device auto-derives from
# the FIRST entry in self.models, which goes CPU after offloading
# image_cond_model and breaks sample_slat's noise creation.
type(pipeline).device = property(lambda self: torch.device("cuda"))

# Load an image
image = Image.open("assets/example_image/T.png")

# Run the pipeline. On 16 GB AMD cards the default pipeline footprint
# (5 submodels resident + sampling activations + decoder output) exceeds
# VRAM. We split the run manually so we can move idle submodels to CPU
# between phases.
formats = os.environ.get('OUTPUT_FORMATS', 'mesh,gaussian').split(',')
formats = [f.strip() for f in formats if f.strip()]
print(f"[example] decoding formats: {formats}")
_gpu_mem("after pipeline.cuda")

with torch.no_grad():
    image = pipeline.preprocess_image(image)
    torch.manual_seed(1)
    cond = pipeline.get_cond([image])
    _gpu_mem("after image cond")

    # Free the image-conditioning model — not needed past this point.
    _move_unused_models_to_cpu(pipeline, keep={
        'sparse_structure_flow_model', 'sparse_structure_decoder',
        'slat_flow_model',
        'slat_decoder_mesh', 'slat_decoder_gs', 'slat_decoder_rf',
    })
    _gpu_mem("after dropping image_cond_model")

    coords = pipeline.sample_sparse_structure(cond, num_samples=1)
    _gpu_mem("after sparse structure sample")

    # Free sparse-structure models — done with them.
    _move_unused_models_to_cpu(pipeline, keep={
        'slat_flow_model',
        'slat_decoder_mesh', 'slat_decoder_gs', 'slat_decoder_rf',
    })
    _gpu_mem("after dropping sparse_structure models")

    slat = pipeline.sample_slat(cond, coords)
    _gpu_mem("after slat sample")

    # Free slat flow model — only decoders left.
    decode_keep = {f'slat_decoder_{f if f != "gaussian" else "gs"}' for f in formats}
    decode_keep = {k.replace('radiance_field', 'rf') for k in decode_keep}
    _move_unused_models_to_cpu(pipeline, keep=decode_keep)
    _gpu_mem("after dropping slat_flow_model + non-decoder models")

    outputs = pipeline.decode_slat(slat, formats=formats)
    _gpu_mem("after decode")

    del slat, cond, coords
    gc.collect()
    torch.cuda.empty_cache()
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# Render the outputs (only those that were decoded).
if 'gaussian' in outputs:
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave("sample_gs.mp4", video, fps=30)
    print("Saved sample_gs.mp4")
    torch.cuda.empty_cache()
if 'radiance_field' in outputs:
    video = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave("sample_rf.mp4", video, fps=30)
    print("Saved sample_rf.mp4")
    torch.cuda.empty_cache()
if 'mesh' in outputs:
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave("sample_mesh.mp4", video, fps=30)
    print("Saved sample_mesh.mp4")
    torch.cuda.empty_cache()

# GLB requires both gaussian and mesh.
if 'gaussian' in outputs and 'mesh' in outputs:
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
    )
    glb.export("sample.glb")
    print("Saved sample.glb")

if 'gaussian' in outputs:
    outputs['gaussian'][0].save_ply("sample.ply")
    print("Saved sample.ply")
