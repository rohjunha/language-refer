mkdir -p ./resources/models/nr3d/
wget -N http://54.201.45.51:5000/lr/nr3d/model.pt
mv model.pt ./resources/models/nr3d
mkdir -p ./resources/models/sr3d/
wget -N http://54.201.45.51:5000/lr/sr3d/model.pt
mv model.pt ./resources/models/sr3d