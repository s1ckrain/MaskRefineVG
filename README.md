# MRVG : Mask Refine Video Generation


Training-free, mask-guided attention for sharper foreground details in video generation.

![pintu-fulicat com-1752662452570](https://github.com/user-attachments/assets/37f2d875-249e-4ab9-beb6-bb493d3122ca)



## Key Features

- **Temporal Attention**: Improves consistency and coherence across video frames by leveraging Freelong’s temporal attention.

- **Mask Attention**: Enables fine-grained control over object-level details through mask-driven attention refinement.

- **Text Prompt Refinement**: Users can specify objects or regions of interest in the text prompt to guide the video regeneration process.

## Pipeline

1. **Initial Video Generation**: Generate a rough video sequence from the user’s text prompt using base model.

2. **Object Localization**: Use GroundingDINO along with user-specified target objects to locate bounding boxes in the initial video frames.

3. **Mask Extraction**: Feed the detected bounding boxes into Segment Anything Model to produce foreground masks for the target objects.

4. **Attention Masking**: Incorporate the foreground masks into the attention mechanism by updating the attention mask.

5. **Video Regeneration**: Regenerate the video from the original text prompt, guided by the refined attention masks to produce a more detailed result.

## Environment Setup

Install the required packages using the provided environment.yaml file:
```sh
conda env create -f environment.yaml
conda activate MRVG
```

## Model Checkpoints

Download pre-trained LaVie models, Stable Diffusion 1.4, stable-diffusion-x4-upscaler to ./pretrained_models. You should be able to see the following:
```txt
├── pretrained_models
│   ├── lavie_base.pt
│   ├── lavie_interpolation.pt
│   ├── lavie_vsr.pt
│   ├── stable-diffusion-v1-4
│   │   ├── ...
└── └── stable-diffusion-x4-upscaler
        ├── ...
```
Follow the official instructions for each model to integrate them into the project(Clone  SAM and  GroundingDINO)

Below is the complete project structure:
```txt
├── MRVG
├── GroundingDINO
├── pretrained_models
├── segment-anything
├── environment.yaml
└── requirements.txt
```

## Quick Start

Specify your text prompt and target objects in configs/sample_mrvg.yaml:
```yaml
text_prompt: [
              "Your descriptions here"
       ] 
grounding_prompt: "Your grounding object"
```
Then run the entire pipeline with:
```sh
bash start.sh
```
