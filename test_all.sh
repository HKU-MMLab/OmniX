
python run_generation.py --prompt "Photorealistic modern living room" --output_dir "outputs/generation_from_text"

python run_generation.py --image "assets/examples/image.png" --prompt "Photorealistic modern living room" --output_dir "outputs/generation_from_image_and_text"

python run_perception.py --panorama "assets/examples/panorama.png" --output_dir "outputs/perception"

python run_all.py --prompt "Photorealistic modern living room" --output_dir "outputs/generation_and_perception_from_text"

python run_all.py --image "assets/examples/image.png" --prompt "Photorealistic modern living room" --output_dir "outputs/generation_and_perception_from_image_and_text"
