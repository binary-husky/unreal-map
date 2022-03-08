import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.cluster.vq import kmeans2
from sklearn.cluster import DBSCAN

def distance_matrix_AB(A, B):
    assert A.shape[-1] == 2 # assert 2D situation
    assert B.shape[-1] == 2 # assert 2D situation
    n_A_subject = A.shape[-2]
    n_B_subject = B.shape[-2]
    A = np.repeat(np.expand_dims(A,-2), n_B_subject, axis=-2) # =>(64, Na, Nb, 2)
    B = np.repeat(np.expand_dims(B,-2), n_A_subject, axis=-2) # =>(64, Nb, Na, 2)
    Bt = np.swapaxes(B,-2,-3) # =>(64, Na, Nb, 2)
    dis = Bt-A # =>(64, Na, Nb, 2)
    dis = np.linalg.norm(dis, axis=-1)
    return dis

class CheatScriptAI():
    def attackers_policy_1(self, attackers_agent, guards_agent):
        '''
        大规模集群协同策略
        Args:
            attackers_agent: 攻击方智能体
            guards_agent: 防守方智能体
        Returns:
        written by qth, 2021/04/13
        '''
        # 初始化
        for i, agent in enumerate(attackers_agent):
            agent.act = np.zeros(self.dim_p)  ## We'll use this now for Graph NN
            agent.can_fire = False

        # 设定我方智能体的感知范围
        obversation_radius =  (self.wall_pos[1] - self.wall_pos[0]) #/3 # 即整个地图x轴方向的1/3
        joint_forces = [0, 0]

        ##############  初始化参数  ###############
        eps = 0.5   # 态势评估时，以该值作为半径，对所有对手分布进行聚类成簇
        MinPoints = 2  # 表示每簇中最少的对手个数
        num_members = 5  # 我方agent每组成员个数
        min_members =3  # 我方agent每组成员最少个数，一旦小于该值，则重新分配组队。

        ################  态势评估  ###############
        # 通过聚类算法，对感知到的对手分布情况进行聚类，评估对手的整体分布情况
        # 聚类的个数为我方agent的分组数
        # 下面先从全局感知角度，利用聚类方法，计算对手分布情况。
        guards_agent_position = np.zeros((len(guards_agent), 2))
        for i_g, guard_agent in enumerate(guards_agent):
            guards_agent_position[i_g] = guard_agent.pos
        # print(guards_agent_position)


        result = DBSCAN(eps, min_samples = MinPoints).fit(guards_agent_position)
        # labels_表示聚类后的类别，通常为0,1,2..,最大数表示聚类后的簇的个数，-1表示属于噪音，
        # 不在要求的范围内的或者个数小于2个的类别都会标识为-1
        # label = result.labels_
        # print(label)
        # cluster_results用于存储落在各个不同簇的元素，其中键值即为簇的类别（0,1,2...)
        cluster_results = {}
        for cluster_index, cluster_class in enumerate(result.labels_):
            if cluster_class not in cluster_results.keys():
                cluster_results[cluster_class] = [guards_agent_position[cluster_index]]
                # print(guards_agent_position[cluster_index])
            else:
                cluster_results[cluster_class].append(guards_agent_position[cluster_index])

        # 对手分布中各簇的中心位置列表，作为我方agent的打击目标
        cluster_centroid = []
        cluster_radius = []
        for key in cluster_results.keys():
            cluster_index_position = np.array(cluster_results[key])
            # print("cluster_class:%d" % key)
            # print(cluster_index_position)
            # 对各个簇的各元素再次聚类，得到该簇的中心点，用于引导我方agent
            # 其中，centroid表示该簇的中心，label表示k-means2方法聚类得到的标签类别，由于聚类k为1
            # 因此，此处的label都是同一类的簇
            # 其中，key=-1的簇不能作为中心，因为所有噪点都标记为-1类，该中心不具有实际意义。
            if key != -1:
                for i in range(5):
                    try:
                        centroid, label  = kmeans2(cluster_index_position, 1, iter=20, minit='++',seed=np.random.randint(100), missing='raise')
                        break
                    except:
                        pass
                    if i >= 4:
                        team_centroid, team_labels = kmeans2(cluster_index_position, 1, iter=20, minit='++',seed=np.random.randint(100), missing = 'warn')
                        print('处理空聚类')
                        break
                        # assert False                
                cluster_centroid.append(centroid)
            else:
                # centroid, label = kmeans2(cluster_index_position, len(cluster_results[key]), iter=20, minit='points')
                # centroid = [centroid]
                for item in cluster_index_position:
                    cluster_centroid.append([item])

            # 根据对手分布情况，即每簇中距离最远的两个个体的距离作为我方的直径。
            dists_to_centroid = np.array(
            [np.linalg.norm(cluster_centroid[-1] - cluster_index_position_pos) for cluster_index_position_pos in cluster_index_position])
            
            cluster_radius.append(max(dists_to_centroid))


        ##################### 对我方agents进行分组 ##################
        # print("num_member4:")
        # print(num_members)
        if self.start_flag:
            # team_centroid_step1聚类中心标准顺序 ；；；teams_result_step1 字典，key=类别，成员=元素，
            self.teams_result_step1, self.team_centroid_step1 = self.get_groups(attackers_agent, num_members)
            self.start_flag = False
        # print(teams_result_step1)
        teams_result_step2 = self.teams_result_step1
        team_centroid_step2 = self.team_centroid_step1
        # 判断各组成员数量是否过少，小于3个时，就需要重新组队
        for key in self.teams_result_step1.keys():
            team_index_agents = np.array(self.teams_result_step1[key])
            if len(team_index_agents) <= min_members:
                # print("num_member5:")
                # print(num_members)
                teams_result_step2, team_centroid_step2 = self.get_groups(attackers_agent, num_members)
                break

        ##################### 对我方agents各分组进行指派 ##################
        target_cluster_centroid = cluster_centroid
        # print("target_cluster_centroid1:")
        # print(target_cluster_centroid)
        if len(cluster_centroid) < len(teams_result_step2):
            num_cluster_centroid = len(cluster_centroid)
            if num_cluster_centroid ==0:
                num_cluster_centroid = 1
            temp_num_members = len(attackers_agent) // num_cluster_centroid
            # print("temp_num_members:")
            # print(temp_num_members)
            # print(cluster_centroid)
            # teams_result_step2, team_centroid_step2 = self.get_groups(attackers_agent, temp_num_members)
            for index in range(len(teams_result_step2) - len(cluster_centroid) ):
                target_cluster_centroid.append(target_cluster_centroid[-1])            
                # print("target_cluster_centroid3:")
            # print(len(teams_result_step2))
            # print(teams_result_step2)
        else:
            target_cluster_centroid = []
            target_cluster_centroid = cluster_centroid[:(len(teams_result_step2))]
            # print("target_cluster_centroid2:")
            # print(target_cluster_centroid)
            # print(cluster_centroid)

        # 此时，先默认k-means计算出来的各聚类中心列表，与聚类列别:0,1,2,3..,-1对应
        # 但上面这个是需要测试的，如果不对应，那么下面这种分配就有问题；
        # print("team_centroid_step2:")
        # print(team_centroid_step2)
        # print("target_cluster_centroid:")
        # 聚类中心标准顺序 -> 计算聚类中心、对方聚类中心矩阵
        # dists_to_target_cluster = np.array(
        #     [[np.linalg.norm(team_centroid_position - target_cluster_centroid_pos) for target_cluster_centroid_pos in target_cluster_centroid]
        #      for i, team_centroid_position in enumerate(team_centroid_step2)])  # 计算距离矩阵
        # 计算距离矩阵
        A = team_centroid_step2
        B = np.array(target_cluster_centroid).squeeze()
        dists_to_target_cluster2 = distance_matrix_AB(A, B)
        # assert (dists_to_target_cluster2 == dists_to_target_cluster).all()
        # print(dists_to_target_cluster)
        # 聚类中心标准顺序 -> 计算聚类中心、对方聚类中心矩阵 -> 匈牙利算法（聚类中心-聚类中心的距离）
        ri_team_to_cluster, ci_team_to_cluster = linear_sum_assignment(dists_to_target_cluster2)
        # 由于下面没有用leader_id,暂时设置一个随机的，后面必须取消。
        # print(len(ci_team_to_cluster))
        self.leader_id = -10
        # print("teams_result_step2:")
        # print(teams_result_step2)
        # team_i = 0
        for key, team_agents in teams_result_step2.items():
            # 我方的每一个聚类 key=类别，team_agents=成员列表，
            # 计算各follower的位置
            # print(team_agents)
            delta_angle = (float)(np.pi / len(team_agents) -1)
            expected_poses_patrol = []
            # print("team_i:%d" % team_i)
            # print("team_class_key:%d" % key)
            # print(teams_result_step2)
            # print(target_cluster_centroid)    # 目标中心（10，2） 对应10个聚类 
            # print(key)
            # print(ci_team_to_cluster[key])    # ci_team_to_cluster （10，1）
            if key >= len(target_cluster_centroid): 
                key = len(target_cluster_centroid) - 1

            target_cluster = ci_team_to_cluster[key]
            leader_position_patrol = np.array(target_cluster_centroid[target_cluster])  # 领航者的位置
            # team_i += 1
            circle_radiu = 0.8
            # circle_radiu = Agent().shootRad * (len(team_agents) - 1) / (2 * np.pi)
            # print(key)
            # circle_radiu = cluster_radius[ci_team_to_cluster[key]] * 0.8

            for i, agent in enumerate(team_agents):
                if agent.iden != self.leader_id:
                    # if fighter_live_num_list[i] > self.leader_id:
                    #     expected_poses_patrol.append(np.array([leader_position_patrol + circle_radiu * np.array(
                    #         [np.cos(angle * (i - 1) * np.pi / 180), np.sin(angle * (i - 1) * np.pi / 180)])]))
                    # else:
                    expected_poses_patrol.append([leader_position_patrol + circle_radiu * np.array(
                        [np.cos(delta_angle * i), np.sin(delta_angle * i)])])
            dists_patrol = np.array(
                [[np.linalg.norm(np.array([agent.pos[0], agent.pos[1]]) - pos) for pos in expected_poses_patrol]
                 for i, agent in enumerate(team_agents) if agent.iden != self.leader_id])
            ri, ci = linear_sum_assignment(dists_patrol)

            for i, agent in enumerate(team_agents):
                if agent.iden == self.leader_id:
                    continue
                # if y == fighter_live_num_list[i]:
                #     if y > self.leader_id:
                #         expected_poses_for_it = expected_poses_patrol[ci[i - 1]]
                #     else:
                # print(expected_poses_patrol)
                expected_poses_for_it = expected_poses_patrol[ci[i]]
                # print("expected_poses_for_it")
                # print(expected_poses_for_it)
                relative_value_patrol = expected_poses_for_it - np.array([agent.pos[0], agent.pos[1]])
                # print(relative_value_patrol)
                theta_patrol = np.arctan2(relative_value_patrol[0][0][1], relative_value_patrol[0][0][0])
                if theta_patrol < 0:
                    theta_patrol += 2 * np.pi

                # print(expected_poses_for_it[0])
                agent.act[0] = -agent.pos[0] + expected_poses_for_it[0][0][0]
                agent.act[1] = -agent.pos[1] + expected_poses_for_it[0][0][1]

                if self.s_cfg.DISALBE_RED_FUNCTION:
                    agent.act[0] = -agent.act[0]
                    agent.act[1] = -agent.act[1]

                # 计算各跟随者的打击角度，超打击范围内，距离最近的对手方向射击
                for guard_i, guard_agent in enumerate(guards_agent):
                    # guard_agent.can_fire = False
                    if (agent.pos[0] - agent.shootRad) <= guard_agent.pos[0] <= (
                            agent.pos[0] + agent.shootRad) and (
                            agent.pos[1] - agent.shootRad) <= guard_agent.pos[1]  <= (
                            agent.pos[1] + agent.shootRad):
                        relative_to_guard = guard_agent.pos - agent.pos
                        theta_patrol = np.arctan2(relative_to_guard[1], relative_to_guard[0])
                        if theta_patrol < 0:
                            theta_patrol += 2 * np.pi
                        # print("attackID:%d, guardID:%d" % (agent.iden,guard_agent.iden))
                if agent.alive:
                    agent.atk_rad = theta_patrol
                agent.can_fire = True
                if self.s_cfg.DISALBE_RED_FUNCTION:
                    agent.can_fire = False


    def get_groups(self, policy_agents, num_members):
        '''
        通过k-means方法对我方agents进行分组
        Args:
            policy_agents: 我方存活的agents
            num_members: 每组要求的最少成员数量

        Returns: 分组后的列表字典，键为分组编号，值为agent类实体。
        written by qth，2021/04/25
        '''

        if num_members == 0:
            num_members = 1
        num_team = len(policy_agents) // num_members
        if num_team == 0:
            num_team = 1
        policy_agents_position = []
        for i, agent in enumerate(policy_agents):
            policy_agents_position.append(agent.pos)
        for i in range(5):
            try:    # its here
                team_centroid, team_labels = kmeans2(policy_agents_position, num_team, iter=20, minit='++',seed=np.random.randint(100), missing = 'raise')
                break
            except:
                pass
            if i >= 4:
                team_centroid, team_labels = kmeans2(policy_agents_position, num_team, iter=20, minit='++',seed=np.random.randint(100), missing = 'warn')
                print('处理空聚类')
                break

        if min(team_labels) != 0:
            assert min(team_labels) == 0
        team_results = {}
        for team_index, team_class in enumerate(team_labels):
            if team_class not in team_results.keys():  
                team_results[team_class] = [policy_agents[team_index]]
            else:
                team_results[team_class].append(policy_agents[team_index])
        return team_results, team_centroid

