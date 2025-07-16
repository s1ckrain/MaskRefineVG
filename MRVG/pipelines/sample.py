import os
import torch
import argparse
# import torchvision

from pipeline_videogen import VideoGenPipeline
from pipeline_regenerate import RegeneratePipeline

from download import find_model
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from omegaconf import OmegaConf

import sys
sys.path.append(os.path.split(sys.path[0])[0])
sys.path.append(os.path.abspath("/root/autodl-tmp/MRVG/segment-anything"))
sys.path.append(os.path.abspath("/root/autodl-tmp/MRVG/GroundingDINO/groundingdino"))
from models import get_models
import imageio
from pathlib import Path
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm
from PIL import Image
import numpy as np

from groundingdino.util.inference import (
    load_model,
    load_image,
    predict    as gd_predict,
    annotate,
)
import cv2

def main(args):


	if args.seed is not None:
		torch.manual_seed(args.seed)
	torch.set_grad_enabled(False)
	device = "cuda" if torch.cuda.is_available() else "cpu"

	sd_path = args.pretrained_path + "/stable-diffusion-v1-4"
	unet = get_models(args, sd_path).to(device, dtype=torch.float16)
	state_dict = find_model(args.ckpt_path)
	unet.load_state_dict(state_dict)
	
	vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float16).to(device)
	tokenizer_one = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
	text_encoder_one = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device) # huge

	# load SAM
	sam = sam_model_registry["vit_h"](checkpoint="/root/autodl-tmp/MRVG/segment-anything/model_ckpt/sam_vit_h_4b8939.pth")
	predictor_sam = SamPredictor(sam)
	sam.to(device = device)

	# load grounding model
	TEXT_PROMPT_FOR_GROUNDING = args.grounding_prompt
	BOX_TRESHOLD = 0.35
	TEXT_TRESHOLD = 0.25
	grounding_model = load_model("/root/autodl-tmp/MRVG/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/root/autodl-tmp/MRVG/GroundingDINO/weights/groundingdino_swint_ogc.pth")

	# set eval mode
	unet.eval()
	vae.eval()
	text_encoder_one.eval()
	
	if args.sample_method == 'ddim':
		scheduler = DDIMScheduler.from_pretrained(sd_path, 
											   subfolder="scheduler",
											   beta_start=args.beta_start, 
											   beta_end=args.beta_end, 
											   beta_schedule=args.beta_schedule)
	elif args.sample_method == 'eulerdiscrete':
		scheduler = EulerDiscreteScheduler.from_pretrained(sd_path,
											   subfolder="scheduler",
											   beta_start=args.beta_start,
											   beta_end=args.beta_end,
											   beta_schedule=args.beta_schedule)
	elif args.sample_method == 'ddpm':
		scheduler = DDPMScheduler.from_pretrained(sd_path,
											  subfolder="scheduler",
											  beta_start=args.beta_start,
											  beta_end=args.beta_end,
											  beta_schedule=args.beta_schedule)
	else:
		raise NotImplementedError

	videogen_pipeline = VideoGenPipeline(vae=vae, 
								 text_encoder=text_encoder_one, 
								 tokenizer=tokenizer_one, 
								 scheduler=scheduler, 
								 unet=unet).to(device)
	videogen_pipeline.enable_xformers_memory_efficient_attention()
	videogen_pipeline.enable_vae_slicing()

	regenerate_pipeline = RegeneratePipeline(vae=vae, 
								 text_encoder=text_encoder_one, 
								 tokenizer=tokenizer_one, 
								 scheduler=scheduler, 
								 unet=unet).to(device)
	regenerate_pipeline.enable_xformers_memory_efficient_attention()
	regenerate_pipeline.enable_vae_slicing()	

	if not os.path.exists(args.output_folder):
		os.makedirs(args.output_folder)

	video_grids = []
	# if args.text_prompt is string or list
	if isinstance(args.text_prompt, str):
		prompts = Path(args.text_prompt).read_text().splitlines()
		if prompts[-1] == '':
			prompts = prompts[:-1]
	else:
		prompts = args.text_prompt

	# generate video for each prompt--the first time 
	for prompt_idx,prompt in enumerate(prompts):
		print('Generating the ({}) prompt'.format(prompt))
		firstgen = videogen_pipeline(prompt, 
								video_length=args.video_length, 
								height=args.image_size[0], 
								width=args.image_size[1], 
								num_inference_steps=args.num_sampling_steps,
								guidance_scale=args.guidance_scale)
		videos = firstgen.video
		# org_latents = firstgen.latents
		imageio.mimwrite(args.output_folder + 'video' + f"{prompt_idx:04d}" + '.mp4', videos[0], fps=8, quality=9) # highest quality is 10, lowest is 0

		frames = videos[0]
		out_mid_dir = os.path.join(args.output_folder, "frames")
		os.makedirs(out_mid_dir, exist_ok=True)
		out_dir = os.path.join(out_mid_dir,f"{prompt_idx:04d}")
		os.makedirs(out_dir, exist_ok=True)

		# save each frame as a PNG
		for idx, frame in enumerate(frames):
			filename = os.path.join(out_dir, f"{idx:04d}.png")
			imageio.imwrite(filename, frame)
		
		video_masks = []
		
		#grounding and segmenting video frames
		for idx in tqdm(range(len(frames)), desc="Grounding & Segmenting video frames", unit="frame"):
			IMAGE_PATH = os.path.join(out_dir, f"{idx:04d}.png")
			image_source, image = load_image(IMAGE_PATH)
			# get the bounding box of the object
			boxes, _, _ = gd_predict(
    			model=grounding_model,
    			image=image,
    			caption=TEXT_PROMPT_FOR_GROUNDING,
    			box_threshold=BOX_TRESHOLD,
    			text_threshold=TEXT_TRESHOLD
			)
			boxes = np.array(boxes) # convert to numpy array
			
			# use SAM to get the mask of the object
			image_seg = imageio.imread(IMAGE_PATH)
			h, w = image_seg.shape[:2]  # get image dimensions
			
			# Convert normalized coordinates to absolute pixel coordinates
			if len(boxes) > 0:
				box = boxes[0]  
				center_x, center_y, box_w, box_h = box
				
				# Convert to absolute coordinates
				x_min = int((center_x - box_w/2) * w)
				y_min = int((center_y - box_h/2) * h)
				x_max = int((center_x + box_w/2) * w)
				y_max = int((center_y + box_h/2) * h)
				
				# Ensure coordinates are within image bounds
				x_min = max(0, x_min)
				y_min = max(0, y_min)
				x_max = min(w, x_max)
				y_max = min(h, y_max)
				
				sam_box = np.array([x_min, y_min, x_max, y_max])
				predictor_sam.set_image(image_seg)
				masks, _, _ = predictor_sam.predict(box=sam_box)
			else:
				print("ERROR:No boxes detected")
				# Create an empty mask if no boxes detected
				masks = [np.zeros((h, w), dtype=bool)]
			video_masks.append(masks[0]) 
		
		print("Successfully get the masks of the object")
		print("Converting masks to attention mask format")
		
		# Convert video_masks to attention_mask format for the regenerate pipeline
		if len(video_masks) > 0:
			# Stack masks into a tensor [T, H, W]
			masks_tensor = torch.from_numpy(np.stack(video_masks)).float()  
			
			# Downsample to latent space resolution (divide by vae_scale_factor=8), Use average pooling to downsample
			T, H, W = masks_tensor.shape
			latent_h, latent_w = H // 8, W // 8
			
			# Reshape and pool
			masks_resized = torch.nn.functional.interpolate(
				masks_tensor.unsqueeze(0).unsqueeze(0),  
				size=(T, latent_h, latent_w),
				mode='trilinear',
				align_corners=False
			).squeeze(0).squeeze(0)  
			
			# Add batch dimension and move to the appropriate device
			attention_mask = masks_resized.unsqueeze(0) 
			print(f"Attention mask shape: {attention_mask.shape}")
		else:
			attention_mask = None
			print("No masks available, proceeding without attention mask")

		print('Regenerating the ({}) prompt'.format(prompt))
		regen = regenerate_pipeline(prompt, 
									video_length=args.video_length, 
									height=args.image_size[0], 
									width=args.image_size[1], 
									num_inference_steps=args.num_sampling_steps,
									guidance_scale=args.guidance_scale,
									attention_mask=attention_mask, )	
		regen_videos = regen.video
		imageio.mimwrite(args.output_folder + 'regen_video' + f"{prompt_idx:04d}" + '.mp4', regen_videos[0], fps=8, quality=9) 

		regen_frames = regen_videos[0]
		out_mid_dir = os.path.join(args.output_folder, "regen_frames")
		os.makedirs(out_mid_dir, exist_ok=True)
		out_dir = os.path.join(out_mid_dir, f"{prompt_idx:04d}")
		os.makedirs(out_dir, exist_ok=True)

		# save each frame as a PNG
		for re_idx, re_frame in enumerate(regen_frames):
			filename = os.path.join(out_dir, f"{re_idx:04d}.png")
			imageio.imwrite(filename, re_frame)

	print("Successfully regenerate the video")
	# print('save path {}'.format(args.output_folder))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="")
	args = parser.parse_args()

	main(OmegaConf.load(args.config))
