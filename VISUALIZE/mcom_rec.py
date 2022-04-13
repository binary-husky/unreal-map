import os
import numpy as np
import matplotlib
from config import GlobalConfig
# 设置matplotlib正常显示中文和负号
# matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
# matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
# plt.ion()
StandardPlotFig = 1
ComparePlotFig = 2
# from pylab import *
class rec_family(object):
    def __init__(self, colorC=None, draw_mode='Native', image_path=None):
        self.name_list = []
        self.line_list = []
        self.line_plot_handle = []
        self.line_plot_handle2 = []
        self.subplots = {}
        self.subplots2 = {}
        self.working_figure_handle = None
        self.working_figure_handle2 = None
        self.smooth_line = False
        self.colorC = 'k' if colorC is None else colorC
        self.Working_path = 'Testing-beta'
        self.image_num = -1
        self.draw_mode = draw_mode
        self.vis_95percent = True
        logdir = GlobalConfig.logdir
        self.plt = None
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if self.draw_mode == 'Web':
            import matplotlib.pyplot as plt, mpld3
            self.html_to_write = '%s/html.html'%logdir
            self.plt = plt; self.mpld3 = mpld3
        elif self.draw_mode =='Native':
            import matplotlib.pyplot as plt
            plt.ion()
            self.plt = plt
        elif self.draw_mode =='Img':
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            self.plt = plt
            self.img_to_write = '%s/rec.jpg'%logdir
            if image_path is not None:
                self.img_to_write = image_path
        else:
            assert False

    def rec_init(self, colorC=None):
        if colorC is not None: self.colorC = colorC
        return

    def rec(self, var, name):
        if name in self.name_list:
            pass
        else:
            self.name_list.append(name)
            self.line_list.append([])  #新建一个列表
            self.line_plot_handle.append(None)
            self.line_plot_handle2.append(None)
        
        index = self.name_list.index(name)
        self.line_list[index].append(var)
    
    # This function is ugly because it is translated from MATLAB
    def rec_show(self):
        image_num = len(self.line_list)  #一共有多少条曲线
        #画重叠曲线，如果有的话
        
        if self.working_figure_handle is None:
            self.working_figure_handle = self.plt.figure(StandardPlotFig, figsize=(12, 6), dpi=100)
            if self.draw_mode == 'Native': 
                self.working_figure_handle.canvas.set_window_title(self.Working_path)
                self.plt.show()
        
        rows = 1
        #检查是否有时间轴，若有，做出修改
        flag_time_e = 0
        encountered = 0 # 有时间轴
        time_index = None
        if 'time' in self.name_list:
            time_index = self.name_list.index('time')
            image_num = image_num - 1
            flag_time_e = 1
            encountered = 0 
        
        if image_num >= 3:
            rows = 2 #大与3张图，则放两行
        if image_num > 8:
            rows = 3 #大与3张图，则放两行
        if image_num > 12:
            rows = 4 #大与3张图，则放两行

        if flag_time_e>0:
            image_num = image_num + 1
        
        cols = int(np.ceil(image_num/rows))#根据行数求列数
        if self.image_num!=image_num:
            # 需要刷新布局，所有已经绘制的图作废
            self.subplots = {}
            self.working_figure_handle.clf()
            for q,handle in enumerate(self.line_plot_handle): 
                self.line_plot_handle[q] = None

        self.image_num = image_num
        self.plot_classic(image_num, rows, flag_time_e, encountered, time_index, cols)
            
        # plt.draw()
        # ##################################################
        # ##################################################
        

        # #画重叠曲线，如果有的话
        draw_advance_fig = False
        for name in self.name_list:
            if 'of=' in name: draw_advance_fig = True

        if draw_advance_fig:
            self.plot_advanced()

        # now end
        self.plt.tight_layout()
        if self.draw_mode == 'Web':
            content = self.mpld3.fig_to_html(self.working_figure_handle)
            with open(self.html_to_write, 'w+') as f:
                f.write(content)
            return
        elif self.draw_mode == 'Native':
            self.plt.pause(0.01)
            return
        elif self.draw_mode == 'Img':
            if self.working_figure_handle is not None: 
                self.working_figure_handle.savefig(self.img_to_write)

    def plot_advanced(self):
        #画重叠曲线，如果有的话
        if self.working_figure_handle2 is None:
            self.working_figure_handle2 = self.plt.figure(ComparePlotFig, figsize=(12, 6), dpi=100)
            if self.draw_mode == 'Native': 
                self.working_figure_handle2.canvas.set_window_title('Working')
                self.plt.show()
        
        group_name = []
        group_member = []
        
        for index in range(image_num):
            if 'of=' not in self.name_list[index]:
                continue    #没有的直接跳过
            # 找出组别
            res = self.name_list[index].split('of=')
            g_name_ = res[0]
            
            if g_name_ in group_name: # any(strcmp(group_name, g_name_)):
                i = group_name.index(g_name_)
                group_member[i].append(index)# ,index]  ##ok<*AGROW>
            else:
                group_name.append(g_name_)
                group_member.append([index])
            
            
        
        num_group = len(group_name)
        image_num_multi = num_group
        if image_num_multi >= 3:
            rows = 2#大与3张图，则放两行
        else:
            rows = 1
        
        cols = int(np.ceil(image_num_multi/rows))#根据行数求列数
        
        for i in range(num_group):

            subplot_index = i+1
            subplot_name = '%d,%d,%d'%(rows,cols,subplot_index)
            if subplot_name in self.subplots2: 
                target_subplot = self.subplots2[subplot_name]
            else:
                target_subplot = self.working_figure_handle2.add_subplot(rows,cols,subplot_index)
                self.subplots2[subplot_name] = target_subplot

            tar_true_name=group_name[i]
            num_member = len(group_member[i])
            
            for j in range(num_member):
                index = group_member[i][j]
                name_tmp = self.name_list[index]
                name_tmp = name_tmp.replace('=',' ')
                if self.smooth_line:
                    target = smooth(self.line_list[index],20) 
                else:
                    target = self.line_list[index]
                if (self.line_plot_handle2[index] is None):
                    if flag_time_e>0:
                        self.line_plot_handle2[index]  =  target_subplot.plot(self.line_list[time_index],self.line_list[index],lw=1,label=name_tmp)
                    else:
                        self.line_plot_handle2[index], =  target_subplot.plot(self.line_list[index], lw=1, label=name_tmp)

                else:
                    if flag_time_e>0:
                        self.line_plot_handle2[index].set_data((self.line_list[time_index],self.line_list[index]))
                    else:
                        xdata = np.arange(len(self.line_list[index]), dtype=np.double)
                        ydata = np.array(self.line_list[index], dtype=np.double)
                        self.line_plot_handle2[index].set_data((xdata,ydata))

            #标题
            target_subplot.set_title(tar_true_name)
            target_subplot.set_xlabel('time')
            target_subplot.set_ylabel(tar_true_name)
            target_subplot.relim()

            limx1 = target_subplot.dataLim.xmin
            limx2 = target_subplot.dataLim.xmax
            limy1 = target_subplot.dataLim.ymin
            limy2 = target_subplot.dataLim.ymax
            # limx1,limy1,limx2,limy2 = target_subplot.dataLim
            if limx1 != limx2 and limy1!=limy2:
                meany = limy1/2 + limy2/2
                limy1 = (limy1 - meany)*1.2+meany
                limy2 = (limy2 - meany)*1.2+meany
                target_subplot.set_ylim(limy1,limy2)
                meanx = limx1/2 + limx2/2
                limx1 = (limx1 - meanx)*1.05+meanx
                limx2 = (limx2 - meanx)*1.05+meanx
                target_subplot.set_xlim(limx1,limx2)
                target_subplot.grid(visible=True)
                target_subplot.legend(loc='best')
            elif limx1 != limx2:
                meanx = limx1/2 + limx2/2
                limx1 = (limx1 - meanx)*1.1+meanx
                limx2 = (limx2 - meanx)*1.1+meanx
                target_subplot.set_xlim(limx1,limx2)

    def plot_classic(self, image_num, rows, flag_time_e, encountered, time_index, cols):
        for index in range(image_num):
            if flag_time_e>0:
                if time_index == index:
                    encountered = 1 # 有时间轴
                    continue
                # 有时间轴时，因为不绘制时间，所以少算一个subplot:
            subplot_index = index if encountered > 0 else index+1
            subplot_name = '%d,%d,%d'%(rows,cols,subplot_index)
            if subplot_name in self.subplots: 
                target_subplot = self.subplots[subplot_name]
            else:
                target_subplot = self.working_figure_handle.add_subplot(rows,cols,subplot_index)
                self.subplots[subplot_name] = target_subplot

            _xdata_ = np.arange(len(self.line_list[index]), dtype=np.double)
            _ydata_ = np.array(self.line_list[index], dtype=np.double)
            if flag_time_e>0:
                _xdata_ = np.array(self.line_list[time_index], dtype=np.double)
                # plt.plot(x,y,ls,lw,c,marker,markersize,markeredgecolor,markerfacecolor,label)
                # **x：**横坐标；**y：**纵坐标；**ls或linestyle：**线的形式（‘-’，‘–’，‘：’和‘-.’）；**lw（或linewidth）：**线的宽度；**c：**线的颜色；**marker：**线上点的形状；**markersize或者ms：**标记的尺寸，浮点型；**markerfacecolor：**点的填充色；**markeredgecolor：标记的边沿颜色label：**文本标签
            if (self.line_plot_handle[index] is None):# || ~isvalid(self.line_plot_handle[index])):
                    if flag_time_e>0:
                        self.line_plot_handle[index]  =  target_subplot.plot(self.line_list[time_index],self.line_list[index],lw=1,c=self.colorC)
                    else:
                        self.line_plot_handle[index], =  target_subplot.plot(self.line_list[index], lw=1, c=self.colorC)
                        
            else:
                if flag_time_e>0:
                    self.line_plot_handle[index].set_data((self.line_list[time_index],self.line_list[index]))
                else:
                    xdata = np.arange(len(self.line_list[index]), dtype=np.double)
                    ydata = np.array(self.line_list[index], dtype=np.double)
                    self.line_plot_handle[index].set_data((xdata,ydata))

            if 'of=' in self.name_list[index]:
                #把等号替换成空格
                name_tmp = self.name_list[index]
                name_tmp = name_tmp.replace('=',' ')
                target_subplot.set_title(name_tmp)
                target_subplot.set_xlabel('time')
                target_subplot.set_ylabel(name_tmp)
                target_subplot.grid(visible=True)
            else:
                target_subplot.set_title(self.name_list[index])
                target_subplot.set_xlabel('time')
                target_subplot.set_ylabel(self.name_list[index])
                target_subplot.grid(visible=True)

            limx1 = _xdata_.min() #target_subplot.dataLim.xmin
            limx2 = _xdata_.max() #target_subplot.dataLim.xmax
            limy1 = _ydata_.min() #min(self.line_list[index])
            limy2 = _ydata_.max() #max(self.line_list[index])

            if len(_ydata_)>220 and self.vis_95percent:
                limy1 = np.percentile(_ydata_, 3, interpolation='midpoint') # 5%
                limy2 = np.percentile(_ydata_, 97, interpolation='midpoint') # 95%

            if limx1 != limx2 and limy1!=limy2:
                    # limx1,limy1,limx2,limy2 = target_subplot.dataLim
                meany = limy1/2 + limy2/2
                limy1 = (limy1 - meany)*1.2+meany
                limy2 = (limy2 - meany)*1.2+meany
                target_subplot.set_ylim(limy1,limy2)

                meanx = limx1/2 + limx2/2
                limx1 = (limx1 - meanx)*1.1+meanx
                limx2 = (limx2 - meanx)*1.1+meanx
                target_subplot.set_xlim(limx1,limx2)
            elif limx1 != limx2:
                meanx = limx1/2 + limx2/2
                limx1 = (limx1 - meanx)*1.1+meanx
                limx2 = (limx2 - meanx)*1.1+meanx
                target_subplot.set_xlim(limx1,limx2)


