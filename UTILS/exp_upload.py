import paramiko, os
from UTILS.colorful import print亮紫, print亮靛
class ChainVar(object):
    def __init__(self, chain_func, chained_with):
        self.chain_func = chain_func
        self.chained_with = chained_with

class DataCentralServer(object): # ADD_TO_CONF_SYSTEM //DO NOT remove this comment//
    addr = 'None'
    usr = 'None'
    pwd = 'None'


# great thank to skoll for sharing this at stackoverflow:
# https://stackoverflow.com/questions/4409502/directory-transfers-with-paramiko
class MySFTPClient(paramiko.SFTPClient):
    def put_dir(self, source, target):
        ''' Uploads the contents of the source directory to the target path. The
            target directory needs to exists. All subdirectories in source are 
            created under target.
        '''
        for item in os.listdir(source):
            if os.path.isfile(os.path.join(source, item)):
                # print亮靛('uploading: %s --> %s'%(os.path.join(source, item),'%s/%s' % (target, item)))
                self.put(os.path.join(source, item), '%s/%s' % (target, item))
            else:
                self.mkdir('%s/%s' % (target, item), ignore_existing=True)
                self.put_dir(os.path.join(source, item), '%s/%s' % (target, item))

    def mkdir(self, path, mode=511, ignore_existing=False):
        ''' Augments mkdir by adding an option to not fail if the folder exists  '''
        try:
            super(MySFTPClient, self).mkdir(path, mode)
        except IOError as e:
            if ignore_existing:
                pass
            else:
                raise

def upload_experiment_results(cfg): # shell it to catch error
    try: upload_experiment_results_(cfg)
    except: pass

def upload_experiment_results_(cfg):
    path = cfg.logdir
    name = cfg.note
    try:
        addr = DataCentralServer.addr     # ssh ubuntu address
        usr = DataCentralServer.usr      # ubuntu user
        pwd = DataCentralServer.pwd      # ubuntu password
        assert addr != 'None' and (addr is not None)
        assert usr != 'None' and (usr is not None)
        assert pwd != 'None' and (pwd is not None)
    except:
        print('No experiment data central server is configured, 没有配置中央日志服务器')
        return

    remote_path = '/home/%s/CenterHmp/'%usr
    ssh = paramiko.SSHClient() 
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
    ssh.connect(addr, username=usr, password=pwd)
    sftp = MySFTPClient.from_transport(ssh.get_transport())
    print亮紫('uploading results: %s --> %s'%(path, '%s/%s'%(remote_path, name)))
    sftp.mkdir(remote_path, ignore_existing=True)
    sftp.mkdir('%s/%s'%(remote_path, name), ignore_existing=True)
    sftp.put_dir(path, '%s/%s'%(remote_path, name))
    sftp.close()
    print亮紫('upload complete')
    