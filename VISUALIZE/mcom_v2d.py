import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
plt.ion()
V2dPlotFig = 3
# import matplotlib.rcsetup as rcsetup; print(rcsetup.all_backends)
'''
['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 
'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
'''
class v2d_family():
    def __init__(self, draw_mode) -> None:
        self.v_name_list = {}
        self.style_list = {}
        self.trival_line_list = {}
        self.trival_line_pair = []
        self.v2d_fig_handle = None
        # self.v_name_list = {  'char_index':{ 'color': ?,'pos':(x,y),'shape':(xxxx,yyyy) }   }
        # self.style_list = {'red': {'style_name_list':[], 'plot_handle':handle} }
        self.draw_mode = draw_mode
        assert draw_mode=='Native', ('only support native')
        pass
    
    def v2d_init(self):
        pass
    
    def v2dx(self, name, xpos, ypos, dir=0, **kargs):
        str = ' '
        split_res = name.split('|')
        char_shape = 'cir'
        char_color = 'k'
        radius = 4
        if len(split_res) >= 1: char_shape = split_res[0]
        if len(split_res) >= 2: char_index = split_res[1]
        if len(split_res) >= 3: char_color = split_res[2]
        if len(split_res) >= 4: radius = float(split_res[3])

        if char_index in self.v_name_list.keys():
            # 曲线已经被注册
            previous_color = self.v_name_list[char_index]['color']
            # 样式发生变化
            if previous_color != char_color:
                # 取消旧的样式
                assert char_index in self.style_list[previous_color]['style_name_list']
                self.style_list[previous_color]['style_name_list'].remove(char_index)
                # 注册新的样式
                self.add_to_style(char_index, char_color)
                self.v_name_list[char_index]['color'] = char_color
        else:
            # 第一次出现
            self.v_name_list[char_index] = {'color':char_color}
            if self.v2d_fig_handle is None:
                self.init_fig()
            self.add_to_style(char_index, char_color)
        # 
        if char_shape == 'cir':
            xc, yc = self.circle_data(0,0,radius,0,360)
        elif char_shape == 'rec':
            xc, yc = self.rec_data(0,0,radius,0)
        elif char_shape == 'tank':
            xc, yc = self.tank_data(0,0,radius,dir, **kargs)

        self.v_name_list[char_index]['pos'] = (xpos, ypos)
        self.v_name_list[char_index]['shape'] = (xc, yc)
        
        
    def init_fig(self):
        self.v2d_fig_handle = plt.figure(V2dPlotFig, figsize=(8, 8), dpi=100)
        # self.v2d_fig_handle.canvas.set_window_title('V2dPlotFig')
        self.v2d_fig_handle.show()
        plt.show(block=False)

    def add_to_style(self, char_index, char_color):
        if char_color not in self.style_list.keys():
            self.style_list[char_color] = {'plot_handle':None,  'style_name_list':[]}
            self.style_list[char_color]['plot_handle'] = self.v2d_fig_handle.gca().plot(0,0, lw=1, c=char_color)[0]
        # 样式已经被注册
        self.style_list[char_color]['style_name_list'].append(char_index)

    def rec_data(self, x,y,r,dir):
        tmp = np.array([[r, 0, -r, 0, r], [0, r, 0, -r, 0]])
        dir = -dir
        tmp = np.array([[np.cos(dir), np.sin(dir)],[-np.sin(dir),np.cos(dir)]]).dot(tmp)
        xp = tmp[0,:]
        yp = tmp[1,:]
        return xp,yp

    def tank_data(self, x,y,r,dir,**kargs):
        x_ = np.array([-0.74, -0.74, -0.55, -0.55, -0.05, -0.05, -0.55, -0.55, -0.55,
       -0.05, -0.05,  0.07,  0.07,  0.07,  0.07,  0.57,  0.57,  0.07,
        0.57,  0.57,  0.75,  0.75,   np.nan, -0.74, -0.74, -0.55, -0.55,
       -0.05, -0.05, -0.55, -0.55, -0.55, -0.05, -0.05,  0.07,  0.07,
        0.07,  0.07,  0.57,  0.57,  0.07,  0.57,  0.57,  0.75,  0.75,
        np.nan])*r
        y_ = np.array([ 0.45, -0.53, -0.53, -0.62, -0.62, -0.42, -0.42, -0.53, -0.42,
            -0.42, -0.53, -0.53, -0.42, -0.53, -0.62, -0.62, -0.42, -0.42,
            -0.42, -0.53, -0.53,  0.45,   np.nan, -0.45,  0.53,  0.53,  0.62,
                0.62,  0.42,  0.42,  0.53,  0.42,  0.42,  0.53,  0.53,  0.42,
                0.53,  0.62,  0.62,  0.42,  0.42,  0.42,  0.53,  0.53, -0.45,
                np.nan])*r
        tx_ = np.array([-0.15, -0.15,  0.45,  0.45, -0.15,  0.45,  0.45,  1.45,  1.45,
                0.45,   np.nan])*r
        ty_ = np.array([-0.14,  0.14,  0.14, -0.14, -0.14, -0.14, -0.04, -0.04,  0.04,
                0.04,   np.nan])*r
        if 'vel_dir' in kargs:
            theta_gun = dir
            theta_ve = kargs['vel_dir']
        else:
            theta_gun = dir
            theta_ve = dir


        tx = tx_*np.cos(theta_gun) - ty_*np.sin(theta_gun)
        ty = tx_*np.sin(theta_gun) + ty_*np.cos(theta_gun)
        x = x_*np.cos(theta_ve) - y_*np.sin(theta_ve)
        y = x_*np.sin(theta_ve) + y_*np.cos(theta_ve)
        xp = np.concatenate((x,tx))
        yp = np.concatenate((y,ty))

        # fan
        if len(kargs)>0:
            fan_r = 1 if kargs is None else kargs['attack_range']
            xfan, yfan = self.fan_data(0,0,fan_r, dir, np.pi/4)
            xp = np.concatenate((xp,xfan))
            yp = np.concatenate((yp,yfan))

        return xp,yp

    def circle_data(self, x,y,r,dir,rad,step=15):
        dir = -dir
        rads = dir - rad/2
        rade = dir + rad/2
        ang = np.arange(start=rads, stop=rade+1e-5, step=step) # rads:45:rade 
        xp=r*np.cos(ang*np.pi/180)+x
        yp=r*np.sin(ang*np.pi/180)+y
        return xp, yp
    
    @staticmethod
    def dotify_vec(p_arr):
        lxp = len(p_arr)
        new_arr_len = lxp + lxp//2 if lxp%2 !=0 else lxp + lxp//2 - 1
        dot_arr = np.arange(new_arr_len)%3
        dot_index=dot_arr + 2*(np.arange(new_arr_len)//3)
        # dot_index[==2] = 0
        p_arr = p_arr[dot_index]; p_arr[dot_arr==2] = np.nan
        return p_arr

    @staticmethod
    def line(p1,p2, sep):
        from UTILS.tensor_ops import repeat_at
        lam = np.arange(start=0.7, stop=1. + 1e-5, step=0.05)
        p1s = repeat_at(p1, insert_dim=0, n_times=len(lam))
        p2s = repeat_at(p2, insert_dim=0, n_times=len(lam))
        lam = repeat_at(lam, insert_dim=-1, n_times=2)
        p_arr_line = p1s*lam + p2s*(1-lam)
        p_arr_line = np.concatenate((p_arr_line, sep),0)
        return p_arr_line


    def fan_data(self, x,y,r,dir,rad):  #to do: dotted line
        rads = dir - rad/2
        rade = dir + rad/2
        ang = np.arange(start=rads, stop=rade+1e-5, step=np.pi/45) # rads:45:rade 
        xp=r*np.cos(ang)+x
        yp=r*np.sin(ang)+y
        sep = np.array([[np.nan,np.nan]])

        p_arr = np.stack((xp,yp)).transpose()
        p_arr = self.dotify_vec(p_arr)
        
        orin = np.array([x,y])

        L1_arr = self.dotify_vec(self.line(p_arr[0], orin, sep))
        L2_arr = self.dotify_vec(self.line(p_arr[-1], orin, sep))
        p_arr = np.concatenate((p_arr, sep),0)

        arr = np.concatenate((L1_arr, p_arr, L2_arr),0)
        return arr[:,0], arr[:,1]

    @staticmethod
    def get_terrain(arr, theta):
        A = 0.05; B=0.2; X=arr[:,0]; Y=arr[:,1]
        X_ = X*np.cos(theta) + Y*np.sin(theta)
        Y_ = -X*np.sin(theta) + Y*np.cos(theta)
        Z = -1 +B*( (0.1*X_) ** 2 + (0.1*Y_) ** 2 )- A * np.cos(2 * np.pi * (0.3*X_))  - A * np.cos(2 * np.pi * (0.5*Y_))
        return -Z

    def v2d_add_terrain(self, theta):
        self.theta = theta
        return

    def v2d_draw(self):
        # self.v_name_list = {  'char_index':{ 'color': ?,'pos':(x,y),'shape':(xxxx,yyyy) }   }
        # self.style_list = {'red': {'style_name_list':[], 'plot_handle':handle} }

        # from UTILS.tensor_ops import my_view
        # X = np.arange(-6, 6, 0.1)
        # Y = np.arange(-6, 6, 0.1)
        # X, Y = np.meshgrid(X, Y)    # 100
        # X = my_view(X, [-1,1])
        # Y = my_view(Y, [-1,1])
        # arr = np.concatenate((X,Y), -1)
        # Z = self.get_terrain(arr, self.theta)
        # d = int(np.sqrt(X.shape[0]))
        # X = X.reshape(d,d)
        # Y = Y.reshape(d,d)
        # Z = Z.reshape(d,d)
        # plt.contourf(X, Y, Z)
        from UTILS.tensor_ops import my_view
        X = np.arange(-6, 6, 0.1)
        Y = np.arange(-6, 6, 0.1)
        X, Y = np.meshgrid(X, Y)    # 100
        X = my_view(X, [-1,1])
        Y = my_view(Y, [-1,1])
        arr = np.concatenate((X,Y), -1)
        Z = self.get_terrain(arr, self.theta)
        d = int(np.sqrt(X.shape[0]))
        X = X.reshape(d,d)
        Y = Y.reshape(d,d)
        Z = Z.reshape(d,d)
        from matplotlib.colors import LinearSegmentedColormap
        cmap_name = 'my_list'
        colors = [(0.4,0.4,0.4),(0.7,0.7,0.7)]
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=10)
        plt.contourf(X, Y, Z, levels= 10,cmap=cmap)

        for style in self.style_list.keys(): 
            style_name_list = self.style_list[style]['style_name_list']
            line_handle = self.style_list[style]['plot_handle']

            x_data_concat = (np.nan,)
            y_data_concat = (np.nan,)

            for char_name in style_name_list:
                xpos, ypos = self.v_name_list[char_name]['pos']
                xc, yc = self.v_name_list[char_name]['shape']
                xc_ = xc + xpos
                yc_ = yc + ypos
                x_data_concat = np.concatenate((x_data_concat, (np.nan,), xc_))
                y_data_concat = np.concatenate((y_data_concat, (np.nan,), yc_)) # [y_data_concat, np.nan, yc_]

            line_handle.set_data((x_data_concat,y_data_concat))
            axes_handle = self.v2d_fig_handle.gca()
            axes_handle.relim()
            axes_handle.axis('equal')
            axes_handle.autoscale_view(True,True,True)
            # self.v2d_fig_handle.gca().set_xlim(-2,2)
            # self.v2d_fig_handle.gca().set_ylim(-2,2)
        for AB in self.trival_line_pair:
            indexA, indexB = AB
            self.v2d_line_object(indexA, indexB)
            
    def v2d_line_object(self, indexA, indexB):
        indexA = str(int(indexA))
        indexB = str(int(indexB))
        line_name = 'line:%s->%s'%(indexA,indexB)
        x1,y1=self.v_name_list[indexA]['pos']
        x2,y2=self.v_name_list[indexB]['pos']
        if line_name not in self.trival_line_list:
            self.trival_line_pair.append([indexA,indexB])
            self.trival_line_list[line_name] = self.v2d_fig_handle.gca().plot([x1,x2],[y1,y2], lw=1, c='k')[0]
        else:
            self.trival_line_list[line_name].set_data(([x1,x2],[y1,y2]))

    def v2d_clear(self):
        self.v_name_list = {}
        self.style_list = {}
        self.trival_line_list = {}
        self.trival_line_pair = []
        if self.v2d_fig_handle is not None:
            self.v2d_fig_handle.clf()


    def v2d_show(self):
        self.v2d_draw()
        plt.draw()
        plt.pause(0.02)
        # print('v2d_show')


if __name__ == '__main__':
    v2d = v2d_family('Native')

    # v2d.v2dx('cir|0|r|0.42', xpos=1, ypos=0)
    # v2d.v2dx('cir|1|r|0.45', xpos=1, ypos=1)
    # v2d.v2dx('rec|2|b|0.1',0,0)
    # v2d.v2dx('rec|0|b|0.1',1,0)
    # plt.pause(15)
    v2d.v2d_init()
    v2d.v2dx('cir|0|b|0.04',1.6974286259771310e+00,5.1136334271362971e+00)
    v2d.v2dx('cir|1|b|0.04',1.7323874630544438e+00,4.8441353012872579e+00)
    v2d.v2dx('cir|2|b|0.04',1.7466729589216059e+00,4.5011681684285119e+00)
    v2d.v2dx('cir|3|b|0.04',1.9834755845632670e+00,4.5303563729102958e+00)
    v2d.v2dx('cir|4|b|0.04',1.7282481584048974e+00,4.2215967719027470e+00)
    v2d.v2dx('cir|5|b|0.04',1.6961323762900244e+00,4.2162281616241906e+00)
    v2d.v2dx('cir|6|b|0.04',1.9538386636567124e+00,3.8049981311503873e+00)
    v2d.v2dx('cir|7|b|0.04',1.8802200761399963e+00,3.6045336461487691e+00)
    v2d.v2dx('cir|8|b|0.04',2.0157571619860644e+00,3.4048652151235701e+00)
    v2d.v2dx('cir|9|b|0.04',1.7335519021352239e+00,3.2864204601228306e+00)
    v2d.v2dx('cir|10|b|0.04',1.7570958563011358e+00,2.9889104070926833e+00)
    v2d.v2dx('cir|11|b|0.04',1.8973304349256952e+00,2.8428589222323897e+00)
    v2d.v2dx('cir|12|b|0.04',1.9387993015459224e+00,2.8276797742219721e+00)
    v2d.v2dx('cir|13|b|0.04',1.9015358683680221e+00,2.3111528346056227e+00)
    v2d.v2dx('cir|14|b|0.04',1.8004736376121440e+00,2.2936583436192697e+00)
    v2d.v2dx('cir|15|b|0.04',2.0107659893617642e+00,2.1508119100878571e+00)
    v2d.v2dx('cir|16|b|0.04',1.8444440354811558e+00,1.9332926354355557e+00)
    v2d.v2dx('cir|17|b|0.04',1.8371937746969322e+00,1.7330254976821911e+00)
    v2d.v2dx('cir|18|b|0.04',1.9129340340533809e+00,1.4154660738691236e+00)
    v2d.v2dx('cir|19|b|0.04',2.0030211676034568e+00,1.4298957885890118e+00)
    v2d.v2dx('cir|20|b|0.04',1.8856683487925392e+00,1.0541625663320517e+00)
    v2d.v2dx('cir|21|b|0.04',2.1527191655945512e+00,1.0831699476943015e+00)
    v2d.v2dx('cir|22|b|0.04',1.8649326792157024e+00,6.2217390250533311e-01)
    v2d.v2dx('cir|23|b|0.04',2.0898862160995813e+00,6.2584505914386446e-01)
    v2d.v2dx('cir|24|b|0.04',1.8549554422225740e+00,2.9299012314167772e-01)
    v2d.v2dx('cir|25|b|0.04',1.9339051071960001e+00,1.7110109106481508e-01)
    v2d.v2dx('cir|26|b|0.04',2.1857357920928222e+00,-2.2904476331429088e-01)
    v2d.v2dx('cir|27|b|0.04',2.2898372430290381e+00,-2.7931351208642319e-01)
    v2d.v2dx('cir|28|b|0.04',2.1052665060702345e+00,-5.2811184015975321e-01)
    v2d.v2dx('cir|29|b|0.04',2.0113388563475842e+00,-5.6067217211506271e-01)
    v2d.v2dx('cir|30|b|0.04',1.9572075843799210e+00,-1.0074934509032205e+00)
    v2d.v2dx('cir|31|b|0.04',2.1019847044222217e+00,-1.2161640572227850e+00)
    v2d.v2dx('cir|32|b|0.04',2.0810334514385351e+00,-1.3997226595437748e+00)
    v2d.v2dx('cir|33|b|0.04',2.1243209345552660e+00,-1.4827073696986688e+00)
    v2d.v2dx('cir|34|b|0.04',2.2812126964925952e+00,-1.6582712682744103e+00)
    v2d.v2dx('cir|35|b|0.04',2.2908731504091189e+00,-2.0073663392281826e+00)
    v2d.v2dx('cir|36|b|0.04',2.1953372997254261e+00,-2.1534518843321200e+00)
    v2d.v2dx('cir|37|b|0.04',2.2671821392880260e+00,-2.2919289063843635e+00)
    v2d.v2dx('cir|38|b|0.04',2.2425046928689309e+00,-2.5855765908809776e+00)
    v2d.v2dx('cir|39|b|0.04',2.1873276051357129e+00,-2.8948300838305192e+00)
    v2d.v2dx('cir|40|b|0.04',2.3101330238715105e+00,-2.7157436369318897e+00)
    v2d.v2dx('cir|41|b|0.04',2.2267736125783628e+00,-3.2590231662938782e+00)
    v2d.v2dx('cir|42|b|0.04',2.1349942847250238e+00,-3.2192762220613687e+00)
    v2d.v2dx('cir|43|b|0.04',2.3555422145111429e+00,-3.4157163022751194e+00)
    v2d.v2dx('cir|44|b|0.04',2.1962188506393736e+00,-3.7758337123121120e+00)
    v2d.v2dx('cir|45|b|0.04',2.3299247859251206e+00,-3.8747477804098480e+00)
    v2d.v2dx('cir|46|b|0.04',2.3401721136660556e+00,-4.0948149979730406e+00)
    v2d.v2dx('cir|47|b|0.04',2.2628131518739241e+00,-4.3771607601792715e+00)
    v2d.v2dx('cir|48|b|0.04',2.2953110060505013e+00,-4.4678125321108535e+00)
    v2d.v2dx('cir|49|b|0.04',2.1851814684351956e+00,-4.6915756064926688e+00)
    v2d.v2d_show()
    v2d.v2dx('cir|100|g|0.04',-2.3340955768552440e+00,4.9801948853894231e+00)
    v2d.v2dx('cir|101|g|0.04',-2.1605749241742314e+00,4.5709127191456567e+00)
    v2d.v2dx('cir|102|g|0.04',-2.0823477095029426e+00,4.4102702646117429e+00)
    v2d.v2dx('cir|103|g|0.04',-2.3605530527535703e+00,4.1945309409909948e+00)
    v2d.v2dx('cir|104|g|0.04',-2.0230254518634014e+00,4.0238393739258154e+00)
    v2d.v2dx('cir|105|g|0.04',-2.3192264590733878e+00,3.8879694623878889e+00)
    v2d.v2dx('cir|106|g|0.04',-2.1216222051303726e+00,3.7463705645806860e+00)
    v2d.v2dx('cir|107|g|0.04',-1.9886601275560660e+00,3.6043159982101840e+00)
    v2d.v2dx('cir|108|g|0.04',-2.1645619676070860e+00,3.1803008917826565e+00)
    v2d.v2dx('cir|109|g|0.04',-2.1347591528027725e+00,3.0688599530093263e+00)
    v2d.v2dx('cir|1010|g|0.04',-2.0320894589688505e+00,2.7911223806467045e+00)
    v2d.v2dx('cir|1011|g|0.04',-2.1154684593180639e+00,2.5239221640697265e+00)
    v2d.v2dx('cir|1012|g|0.04',-2.0346350044844881e+00,2.5034762252365295e+00)
    v2d.v2dx('cir|1013|g|0.04',-2.0529558583644394e+00,2.1800419006883898e+00)
    v2d.v2dx('cir|1014|g|0.04',-2.0592568875840254e+00,1.8981848761033615e+00)
    v2d.v2dx('cir|1015|g|0.04',-2.1020681290870282e+00,1.6573285174084387e+00)
    v2d.v2dx('cir|1016|g|0.04',-2.0784374145153635e+00,1.5212721940167344e+00)
    v2d.v2dx('cir|1017|g|0.04',-2.0408300876939038e+00,1.2753795759495554e+00)
    v2d.v2dx('cir|1018|g|0.04',-2.0044318724067858e+00,1.1335091565257751e+00)
    v2d.v2dx('cir|1019|g|0.04',-1.9962920458871403e+00,1.0817376787031874e+00)
    v2d.v2dx('cir|1020|g|0.04',-1.9654666619042238e+00,8.6786883386340830e-01)
    v2d.v2dx('cir|1021|g|0.04',-1.9629752482380298e+00,7.2412441693175056e-01)
    v2d.v2dx('cir|1022|g|0.04',-1.9827771806940886e+00,4.1249121462499094e-01)
    v2d.v2dx('cir|1023|g|0.04',-1.8148178315400298e+00,3.0620976611163397e-01)
    v2d.v2dx('cir|1024|g|0.04',-1.9913627283935029e+00,-6.7511623558157152e-04)
    v2d.v2dx('cir|1025|g|0.04',-1.9085102331050483e+00,-1.2091692519257527e-01)
    v2d.v2dx('cir|1026|g|0.04',-1.7567025512845775e+00,-4.0655182374325061e-01)
    v2d.v2dx('cir|1027|g|0.04',-1.7573126091984348e+00,-4.7886369068066464e-01)
    v2d.v2dx('cir|1028|g|0.04',-1.7946528244114961e+00,-7.4632155601468431e-01)
    v2d.v2dx('cir|1029|g|0.04',-2.0154230209100197e+00,-9.3702290308563940e-01)
    v2d.v2dx('cir|1030|g|0.04',-1.8071068705401963e+00,-1.2335533997391062e+00)
    v2d.v2dx('cir|1031|g|0.04',-1.7306796885227855e+00,-1.4972874453342571e+00)
    v2d.v2dx('cir|1032|g|0.04',-1.9800566158586790e+00,-1.5443765055388390e+00)
    v2d.v2dx('cir|1033|g|0.04',-1.7202354172897882e+00,-1.7671372514658545e+00)
    v2d.v2dx('cir|1034|g|0.04',-1.8391528999004707e+00,-2.0335429832622496e+00)
    v2d.v2dx('cir|1035|g|0.04',-1.8519982797134462e+00,-2.1369212711162078e+00)
    v2d.v2dx('cir|1036|g|0.04',-1.6969571202242850e+00,-2.3581547587302154e+00)
    v2d.v2dx('cir|1037|g|0.04',-1.6711310745287675e+00,-2.5002987709398945e+00)
    v2d.v2dx('cir|1038|g|0.04',-1.8271627532433052e+00,-2.7791709457111011e+00)
    v2d.v2dx('cir|1039|g|0.04',-1.7163698381455172e+00,-3.0549299320603716e+00)
    v2d.v2dx('cir|1040|g|0.04',-1.8300183826082943e+00,-3.0255021281117158e+00)
    v2d.v2dx('cir|1041|g|0.04',-1.7901509034098166e+00,-3.1658675039835322e+00)
    v2d.v2dx('cir|1042|g|0.04',-1.7876326339702986e+00,-3.5312100308804002e+00)
    v2d.v2dx('cir|1043|g|0.04',-1.6117960847909396e+00,-3.8080053094967141e+00)
    v2d.v2dx('cir|1044|g|0.04',-1.7178237689728504e+00,-4.0787409605324969e+00)
    v2d.v2dx('cir|1045|g|0.04',-1.6671618187791717e+00,-4.2900765366717764e+00)
    v2d.v2dx('cir|1046|g|0.04',-1.6222484609708783e+00,-4.3662799716158425e+00)
    v2d.v2dx('cir|1047|g|0.04',-1.7738583714976750e+00,-4.4039107431128137e+00)
    v2d.v2dx('cir|1048|g|0.04',-1.7155332780077144e+00,-4.8044841550151869e+00)
    v2d.v2dx('cir|1049|g|0.04',-1.7551213302728401e+00,-5.0045607725649397e+00)
    v2d.v2d_show()
    time.sleep(5)
    input("enter omega: ")