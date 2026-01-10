cd /root/flood
mkdir data
cd ./data
mkdir opsfloodnet
cd ./opsfloodnet
mkdir images
mkdir labels
unzip /root/flood/floodnet.zip
mv -f /root/flood/floodnet/train/images/ /root/flood/data/opsfloodnet/images/train/
mv -f /root/flood/floodnet/val/val-org-img/ /root/flood/data/opsfloodnet/images/val/
mv -f /root/flood/floodnet/test/test-org-img/ /root/flood/data/opsfloodnet/images/test/

mv -f /root/flood/floodnet/train/labels/ /root/flood/data/opsfloodnet/labels/train/
mv -f /root/flood/floodnet/val/val-label-img/ /root/flood/data/opsfloodnet/labels/val/
mv -f /root/flood/floodnet/test/test-label-img/ /root/flood/data/opsfloodnet/labels/test/


