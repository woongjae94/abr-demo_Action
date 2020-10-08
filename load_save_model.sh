curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=12TbcnswEFB---mYzisGLiF92mnVCP2E_" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=12TbcnswEFB---mYzisGLiF92mnVCP2E_" -o save_model.tar.gz
tar -xvzf *.tar.gz
rm -rf *.tar.gz
