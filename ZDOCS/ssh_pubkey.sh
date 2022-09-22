key_name=notebook_old
wget --user=fuqingxu   --password=clara http://cloud.fuqingxu.top:4080/remote.php/dav/files/fuqingxu/keys/$key_name.pub -O ./TEMP/_xkey
cat  ./TEMP/_xkey >>  ~/.ssh/authorized_keys 
cat  ~/.ssh/authorized_keys
