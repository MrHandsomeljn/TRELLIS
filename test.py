import os
import sys
import argparse

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.utils import insertany3d_render_utils



parser = argparse.ArgumentParser(description='TRELLIS Image to 3D Pipeline')
parser.add_argument('input_image', help='输入图像路径')
parser.add_argument('--save_base'  , help='输出目录路径'              , default="./output")
parser.add_argument('--save_name'  , help='保存文件名（不包含扩展名）', default=None      )
parser.add_argument('--skip_glb' , action='store_true', help='跳过GLB文件生成')
parser.add_argument('--skip_nerf', action='store_true', help='跳过GLB文件生成')
args = parser.parse_args()

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

# Load an image
image = Image.open(args.input_image)

# Run the pipeline
formats = ['gaussian']
if not args.skip_glb : format += ["mesh"]
if not args.skip_nerf: format += ["radiance_field"]

outputs = pipeline.run(
    image,
    seed=1,
    formats=formats
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 3,
    # },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# Render the outputs

insertany3d_render_utils.save_gaussian_result(outputs['gaussian'][0], args.save_base, args.save_name)

print("Rendering Gaussian:")
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave("sample_gs.mp4", video, fps=30)

if not args.skip_nerf:
    print("Rendering Radiance Field:")
    video = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave("sample_rf.mp4", video, fps=30)

if not args.skip_glb:
    print("Rendering Mesh:")
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave("sample_mesh.mp4", video, fps=30)
    glb_path = f"{args.save_base}/sample.glb"
    print(f"Extracting Mesh to GLB <glb_path>:")
    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(glb_path)
    
print("Finished.")