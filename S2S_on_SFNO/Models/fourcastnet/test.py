from models import FourCastNet1

kwargs={'download_assets':True, 
        'input':'cds',
        'output':'file',
        'path':'/home/lenny/Uni/Master/ai-models-fourcastnet/Output',
        'metadata':{},
        'model_args' : None,
        'assets' : '/home/lenny/Uni/Master/ai-models-fourcastnet/Assets',
        'assets_sub_directory' : None,
        'date': "-1",
        'time': 12,
        'lead-time':240
        }

FCN = FourCastNet1(True,**kwargs) #"owner"
FCN.run()
