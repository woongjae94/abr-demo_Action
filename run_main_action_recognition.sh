sudo docker run --gpus '"device=1"' --link cam_container -v ~/Desktop/abr-demo_Action:/home/app -it --rm --name Action_container woongjae94/abr-demo:Action
