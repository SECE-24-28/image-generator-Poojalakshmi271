from diffusers import StableDiffusionPipeline
import torch
import os


def generate_image(prompt):
    os.makedirs("result", exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )

    image = pipe(prompt).images[0]

    output_path = "result/generated_image.png"
    image.save(output_path)

    print(f"\nImage saved successfully!")
    print(f"Location: {output_path}")


if __name__ == "__main__":
    prompt = input("Enter image description: ")
    generate_image(prompt)
