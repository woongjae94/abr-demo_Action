curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=12TbcnswEFB---mYzisGLiF92mnVCP2E_" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=12TbcnswEFB---mYzisGLiF92mnVCP2E_" -o save_model.tar.gz

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1sl7H2zOy4i30qmtCXlaxO4jBOytIw0mj" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1sl7H2zOy4i30qmtCXlaxO4jBOytIw0mj" -o darknet.tar.gz
tar -xvzf save_model.tar.gz
tar -xvzf darknet.tar.gz
rm -rf *.tar.gz
