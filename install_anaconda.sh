export CONTREPO=https://repo.continuum.io/archive/
export ANACONDAURL=$(wget -q -O - $CONTREPO index.html | grep "Anaconda3-" | grep "Linux" | grep "86_64" | head -n 1 | cut -d \" -f 2)
wget -O ~/anaconda.sh $CONTREPO$ANACONDAURL
bash ~/anaconda.sh -b && rm ~/anaconda.sh && echo '# added by Anaconda3 installer' >> ~/.bashrc && echo '. $HOME/anaconda3/etc/profile.d/conda.sh' >> ~/.bashrc