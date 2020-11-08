sudo docker run --gpus '"device=1"' --network="host" -v ~/Desktop/abr-demo_Action:/home/app -it --rm --name Action_container woongjae94/abr-demo:Action
