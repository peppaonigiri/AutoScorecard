import sys, os
import pandas as pd
import numpy as np
import zipfile
import json
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')

class TzScore:
    """
    评分模型的打分工具类
    
    该类用于对数据进行模型打分，支持以下功能：
    1. 单模型打分：对每个数据源和客群分别使用对应的单模型进行打分
    2. 融合模型打分：基于单模型的概率输出，使用融合模型进行二次打分
    
    工作流程：
    1. 读取压缩的数据文件(.txt.zip)
    2. 加载训练好的 XGBoost 模型(.json)
    3. 进行预测并计算概率值
    4. 根据打分参数将概率转换为评分卡分数
    5. 保存打分结果
    
    参数说明：
        label (pd.DataFrame): 包含待打分用户基础信息的标签数据，必须包含 ex_lst 中指定的关键列
        ex_lst (list): 关键列列表，用于数据去重和表连接，通常为 ['mobile', 'backPointTime'] 或其他唯一标识列组合
        cust_list (list): 客群列表，例如 ['api', 'app', 'xxl', 'all']，对应不同的用户群体
        data_list (list): 数据源名称列表，对应模型文件夹的命名，例如 ['deltaV1', 'thetaV2', 'zetaV2']
        feature_list (list): 融合模型中的特征前缀列表，对应融合模型的变量命名，例如 ['delta', 'theta', 'zeta']
        file_data_list (list): 待打分文件的命名列表，对应实际数据文件的命名，例如 ['deltaV1', 'thetaV2', 'zetaV2']
        file_base_name (str): 数据文件的基础名称前缀，例如 'rz'，实际文件名为 'rz_deltaV1.txt.zip'
        ronghe_list (list): 融合模型名称列表，例如 ['ronghe']
        model_report_path (str): 模型报告和模型文件的根目录路径
        data_or_path (str): 待打分数据文件所在的目录路径
        save_path (str): 打分结果的保存路径
        debug (bool): 调试模式，True 时只读取每个数据文件的前100行，用于快速测试
        
    设计说明：
        - ex_lst 参数化设计：允许灵活指定用于去重和连接的关键列，适应不同的业务场景
        - 例如可以是 ['mobile', 'backPointTime'] 或 ['user_id', 'apply_date'] 等组合
    """
    def __init__(self, label, ex_lst, cust_list, data_list, feature_list, file_data_list, file_base_name, ronghe_list, model_report_path, data_or_path, save_path, debug=False):
        self.debug=debug
        print(f'debug: {debug}')
        self.model_report_path = model_report_path
        self.data_or_path = data_or_path
        self.cust_list = cust_list
        self.data_list = data_list
        self.feature_list = feature_list
        self.file_data_list = file_data_list
        self.file_base_name = file_base_name
        self.ex_lst = ex_lst
        self.ronghe_list = ronghe_list
        self.label = label
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.data_map = self.get_data_map() # 这个map是针对文件名与融合模型中的特征名字命名的差异，融合模型中的特征名一般都是不带v的小写
        self.file_data_map = self.get_file_data_map() # 这个map是针对文件名与待打分的文件的命名方式

    def get_data_map(self):
        """
        构建融合模型特征名到数据源名的映射字典
        
        用于融合模型打分时的特征名转换。融合模型中的特征通常使用简化的命名（如 'delta_xxx_proba'），
        需要映射回完整的数据源名称（如 'deltaV1'）来查找对应的列。
        
        返回:
            dict: 融合特征前缀到数据源名的映射，例如 {'delta': 'deltaV1', 'theta': 'thetaV2', 'zeta': 'zetaV2'}
        """
        return dict(zip(self.feature_list, self.data_list))
        # # 这个处理是基于融合模型的变量名的规范性，避免手动输入融合模型中的变量名采用的处理方式
        # houzhui_list = [f'v{i}' for i in range(1,10)] + [f'V{i}' for i in range(1,10)]
        # deal_data_list = []
        # for data in self.data_list:
        #     for houzhui in houzhui_list:
        #         if houzhui in data:
        #             deal_data_list.append(data.replace(houzhui,'').lower())
        #             break
        #         else:
        #             if houzhui == houzhui_list[-1]:
        #                 deal_data_list.append(data.lower())
        # return dict(zip(deal_data_list,self.data_list))

    def get_file_data_map(self):
        """
        构建数据源名到文件命名的映射字典
        
        用于根据数据源名查找实际的数据文件名。当数据源名称与实际文件命名不一致时需要此映射。
        
        返回:
            dict: 数据源名到文件名的映射，例如 {'deltaV1': 'deltaV1', 'thetaV2': 'thetaV2'}
        """
        return dict(zip(self.data_list,self.file_data_list))

    @staticmethod
    def card_score(pred, P=0, Q=0, PDO=0, good_weight=1):
        """
        将预测概率转换为评分卡分数
        
        使用标准的评分卡公式: Score = A - B * ln(Odds)
        其中: A = P + B * ln(Q), B = -PDO / ln(2)
              Odds = (1-pred) * good_weight / pred
        
        参数:
            pred (np.array): 模型预测的违约概率（坏样本概率）
            P (float): 基准分数，通常为某个整数如 600
            Q (float): 基准 Odds 值，通常为 50（表示好坏比为50:1）
            PDO (float): Points to Double the Odds，分数翻倍时 Odds 的变化幅度，通常为 20-50
            good_weight (float): 好样本权重，用于调整 Odds 计算，默认为1
        
        返回:
            np.array: 评分卡分数（整数），分数越高表示风险越低
        
        示例:
            >>> proba = np.array([0.1, 0.3, 0.5])
            >>> scores = TzScore.card_score(proba, P=600, Q=50, PDO=20)
            >>> print(scores)  # 输出类似 [700, 620, 580]
        """
        Odds = (1-pred)*good_weight/pred
        B = -PDO/np.log(2)
        A = P + B*np.log(Q)
        score = A - B*np.log(Odds)
        # score[score<350] = 350
        # score[score>850] = 1000
        return score.astype(int)
    
    @staticmethod
    def get_score_params(model_report_path,data, cust):
        """
        从模型评估报告中读取打分参数
        
        从对应模型的 Excel 报告文件中读取 P、Q、PDO 等评分卡参数。
        
        参数:
            model_report_path (str): 模型报告根目录路径
            data (str): 数据源名称，如 'deltaV1'
            cust (str): 客群名称，如 'api'
        
        返回:
            dict: 包含打分参数的字典，格式为 {'P': 600, 'Q': 50, 'PDO': 20}
        
        文件路径示例:
            {model_report_path}/deltaV1_api/模型评估报告.xlsx (sheet_name='打分参数')
        """
        report_file = os.path.join(model_report_path, f'{data}_{cust}', '模型评估报告.xlsx')
        df = pd.read_excel(report_file, sheet_name='打分参数')
        return df[['P','Q','PDO']].to_dict(orient='records')[0]

    def score_main(self,data,cust,debug=False,if_only_proba=False):
        """
        单模型打分主函数
        
        执行一个特定数据源和客群组合的打分流程：
        1. 加载模型文件和特征列表
        2. 从压缩文件中读取待打分数据
        3. 去重处理（基于 mobile 和 backPointTime）
        4. 使用模型预测并计算概率
        5. 将概率转换为评分卡分数（可选）
        
        参数:
            data (str): 数据源名称，如 'deltaV1'
            cust (str): 客群名称，如 'api'
            debug (bool): 是否为调试模式，True 时只读取100条数据
            if_only_proba (bool): 是否只输出概率值不计算分数，用于融合模型的输入
        
        返回:
            pd.DataFrame: 包含以下列的数据框
                - mobile: 手机号（或MD5值）
                - backPointTime: 回溯时间点
                - {data}_{cust}_proba: 预测概率
                - {data}_{cust}_score: 评分卡分数（if_only_proba=False 时）
        
        文件路径示例:
            模型文件: {model_report_path}/deltaV1_api/上线文档/model.json
            数据文件: {data_or_path}/rz_deltaV1.txt.zip
        """
        model_file = os.path.join(self.model_report_path, f'{data}_{cust}', '上线文档/model.json')
        with open(model_file, 'r', encoding='utf-8') as f:
            model_json = json.load(f)
        keep_lst = model_json['learner']['feature_names']

        data_filename = f'{self.file_base_name}_{self.file_data_map[data]}.txt.zip'
        with zipfile.ZipFile(self.data_or_path + data_filename, 'r') as zip_file:
            file_list = zip_file.namelist()
            print("压缩包内文件:", file_list)
            csv_filename = [f for f in file_list if f.endswith('.txt') or f.endswith('.csv')][0]
            with zip_file.open(csv_filename) as csv_file:
                if debug:
                        df = pd.read_csv(csv_file, nrows=100,usecols=keep_lst+self.ex_lst)
                else:
                    df = pd.read_csv(csv_file,usecols=keep_lst+self.ex_lst)

        score_params = self.get_score_params(self.model_report_path,data,cust)

        model = xgb.XGBClassifier()
        model.load_model(model_file)

        # if data == 'AlphaV3':
        #     object_columns = []
        #     for column in df.select_dtypes('O').columns:
        #         if column not in ['name','idCard','mobile','reqToken','backPointTime']:
        #             # print(column)
        #             object_columns.append(column)
        #     with open(model_path + 'AlphaV3_mapping_dict.json', 'r', encoding='utf-8') as f:
        #         mapping_dict = json.load(f)
        #     df = apply_categorical_mapping(df,mapping_dict,object_columns)
        
        print(df.shape)
        df.drop_duplicates(self.ex_lst,inplace=True)
        print(df.shape)
        df[f'{data}_{cust}_proba'] =  model.predict_proba(df[keep_lst].fillna(-999))[:, 1]
        if if_only_proba:
            df = df[self.ex_lst + [f'{data}_{cust}_proba']]
        else:
            df[f'{data}_{cust}_score'] = self.card_score(df[f'{data}_{cust}_proba'], **score_params)
            df = df[self.ex_lst + [f'{data}_{cust}_score',f'{data}_{cust}_proba']]
        if 'backPointTime' in df.columns:
            df['backPointTime'] = pd.to_datetime(df['backPointTime'])
        return df

    def score_main_ronghe(self,data,cust,df):
        """
        融合模型打分主函数
        
        使用融合模型对已经过单模型打分的数据进行二次打分。融合模型的输入特征是各个单模型的概率输出。
        
        工作流程：
        1. 加载融合模型文件和特征列表
        2. 将融合模型中的特征名（如 'delta_api_proba'）映射到实际列名（如 'deltaV1_api_proba'）
        3. 使用映射后的数据进行预测
        4. 计算融合模型的概率和评分卡分数
        
        参数:
            data (str): 融合模型名称，如 'ronghe'
            cust (str): 客群名称，如 'api'
            df (pd.DataFrame): 已包含单模型概率列的数据框，必须包含所有融合模型需要的特征列
        
        返回:
            pd.DataFrame: 添加了融合模型概率和分数列的数据框
                新增列:
                - {data}_{cust}_proba: 融合模型预测概率
                - {data}_{cust}_score: 融合模型评分卡分数
        
        特征映射示例:
            融合模型特征 'delta_api_proba' -> 实际列名 'deltaV1_api_proba'
        """
        model_file = os.path.join(self.model_report_path, f'{data}_{cust}', '上线文档/model.json')
        with open(model_file, 'r', encoding='utf-8') as f:
            model_json = json.load(f)
        keep_lst = model_json['learner']['feature_names']

        # 这里是因为考虑到待打分的文件的命名，即data_list中的命名与模型参数中的命名存在差异，模型参数中的命名默认为全部小写的数据名称
        keep_lst_new = [self.data_map[i.split('_')[0]] + f'_{cust}_proba' for i in keep_lst]
        
        score_params = self.get_score_params(self.model_report_path,data,cust)

        model = xgb.XGBClassifier()
        model.load_model(model_file)

        print(df.shape)
        df[f'{data}_{cust}_proba'] =  model.predict_proba(df[keep_lst_new].rename(columns=dict(zip(keep_lst_new,keep_lst))).fillna(-999))[:, 1]
        df[f'{data}_{cust}_score'] = self.card_score(df[f'{data}_{cust}_proba'], **score_params)

        return df

    def single_model_score(self):
        """
        批量执行所有单模型打分
        
        遍历所有数据源和客群的组合，依次调用 score_main 方法进行打分，
        并将打分结果通过 left join 合并到 label 数据框中。
        
        处理流程:
        1. 双重循环遍历 data_list × cust_list 的所有组合
        2. 对每个组合调用 score_main 获取打分结果
        3. 基于 self.ex_lst 指定的列进行左连接合并（通常为 ['mobile', 'backPointTime']）
        4. 保存最终结果为 parquet 文件
        
        输出文件:
            {save_path}/label_single_model.parquet
        
        示例:
            若 data_list=['deltaV1', 'thetaV2'], cust_list=['api', 'app']
            则会生成 4 组打分结果: deltaV1_api, deltaV1_app, thetaV2_api, thetaV2_app
        """
        for data in self.data_list:
            for cust in self.cust_list:
                tmp = self.score_main(data,cust,debug=self.debug,if_only_proba=False)
                self.label = pd.merge(self.label,tmp,how='left',on=self.ex_lst)
        self.label.to_parquet(self.save_path + f'label_single_model.parquet')
        print('=='*30,'子模型打分结果已保存','=='*30)

    def ronghe_model_score(self):
        """
        批量执行所有融合模型打分
        
        基于单模型的打分结果，使用融合模型进行二次打分。融合模型以各单模型的概率值作为输入特征。
        
        注意:
        - 必须先执行 single_model_score() 方法，确保 self.label 中包含所有需要的单模型概率列
        - 融合模型的特征通常是 '{data}_{cust}_proba' 格式的概率值
        
        处理流程:
        1. 遍历融合模型列表和客群列表
        2. 对每个组合调用 score_main_ronghe 获取融合打分
        3. 直接在 self.label 上添加融合模型的概率和分数列
        4. 保存最终结果为 parquet 文件
        
        输出文件:
            {save_path}/rz_label_all_model.parquet
        """
        for ronghe in self.ronghe_list:
            for cust in self.cust_list:
                self.label = self.score_main_ronghe(ronghe, cust, self.label)
        self.label.to_parquet(self.save_path + 'rz_label_all_model.parquet')
        print('=='*30,'融合模型打分结果已保存','=='*30)
    
    def get_score(self):
        """
        完整打分流程的入口方法
        
        按顺序执行单模型打分和融合模型打分，返回包含所有打分结果的数据框。
        
        执行顺序:
        1. single_model_score(): 执行所有单模型打分
        2. ronghe_model_score(): 基于单模型结果执行融合模型打分
        
        返回:
            pd.DataFrame: 包含所有模型打分结果的完整数据框
                - 原始标签列: mobile, backPointTime 等
                - 单模型列: {data}_{cust}_score, {data}_{cust}_proba
                - 融合模型列: {ronghe}_{cust}_score, {ronghe}_{cust}_proba
        """
        self.single_model_score()
        self.ronghe_model_score()
        return self.label

if __name__ == '__main__':
    """
    使用示例：批量模型打分
    
    场景说明:
    假设我们有 3 个数据源（deltaV1, thetaV2, zetaV2）和 4 个客群（api, app, xxl, all），
    需要对这些组合分别进行打分，然后使用融合模型进行二次打分。
    
    文件结构要求:
    1. 模型文件:
       {model_report_path}/{data}_{cust}/上线文档/model.json    # 模型文件
       {model_report_path}/{data}_{cust}/模型评估报告.xlsx      # 打分参数
    
    2. 数据文件:
       {data_or_path}/{file_base_name}_{data}.txt.zip          # 待打分数据（压缩格式）
    
    3. 标签文件:
       {data_or_path}/rz_50w_20240501_20241230_fpd30_dpd30.csv # 用户基础信息
    
    输出文件:
       {save_path}/label_single_model.parquet  # 单模型打分结果
       {save_path}/rz_label_all_model.parquet  # 包含融合模型的完整打分结果
    """
    
    # ==================== 参数配置 ====================
    
    debug = True  # 调试模式：True 时每个数据源只读取100条记录，False 时读取全部数据
    
    # 客群列表：定义不同的用户群体类型
    cust_list = ['api','app','xxl','all']
    # - api: API渠道用户
    # - app: APP渠道用户  
    # - xxl: 信销类用户
    # - all: 所有用户
    
    # 关键列列表：用于数据去重和表连接的关键字段
    # 这些列必须在标签数据和待打分数据中都存在，且能唯一标识或区分记录
    ex_lst = ['mobile', 'backPointTime']
    # - mobile: 用户唯一标识（手机号MD5或明文）
    # - backPointTime: 回溯时间点，用于时间维度的区分

    # 数据源列表：对应模型报告文件夹的命名（用于定位模型文件和报告）
    data_list = ['deltaV1','thetaV2','zetaV2']
    
    # 融合模型特征前缀：对应融合模型中的变量命名（通常是简化版）
    feature_list = ['delta','theta','zeta']
    # 注意：feature_list 与 data_list 的对应关系至关重要
    # 例如：融合模型中的 'delta_api_proba' 会映射到 'deltaV1_api_proba'
    
    # 待打分文件的命名：对应实际数据文件的命名
    file_data_list = ['deltaV1','thetaV2','zetaV2']
    
    # 融合模型列表：定义使用哪些融合模型
    ronghe_list = ['ronghe']
    
    # 数据文件基础名称：实际文件名格式为 {file_base_name}_{data}.txt.zip
    file_base_name = 'rz'
    # 例如: rz_deltaV1.txt.zip, rz_thetaV2.txt.zip
    
    # ==================== 路径配置 ====================
    
    # 模型报告根目录：存放所有模型的评估报告和模型文件
    model_report_path = r'D:\work\06.std_product\xyf\all results\xyf_better/'
    
    # 数据文件目录：存放待打分的压缩数据文件和标签文件
    data_or_path = r'D:\work\06.std_product\918_std_product\rz/'
    
    # 结果保存目录：打分结果的输出路径
    save_path = r'D:\work\06.std_product\xyf\tmp_score_test/'
    
    # ==================== 加载标签数据 ====================
    
    # 读取包含用户基础信息的标签文件
    label = pd.read_csv(data_or_path + r'\rz_50w_20240501_20241230_fpd30_dpd30.csv')
    
    # 数据预处理：将申请日期转换为日期格式
    label['app_dt'] = pd.to_datetime(label['app_dt'].astype('int').astype('str'))
    
    # 列名统一化：将列名重命名为模型所需的标准格式
    label.rename(columns={'mobile_md5':'mobile','app_dt':'backPointTime'},inplace=True)
    # - mobile: 用户唯一标识（手机号MD5或明文）
    # - backPointTime: 回溯时间点，用于与打分数据进行匹配
    

    # ==================== 执行打分 ====================
    
    # 1. 初始化打分类
    scorer = TzScore(
        label=label,                          # 用户标签数据
        ex_lst=ex_lst,                        # 关键列列表（用于去重和连接）
        cust_list=cust_list,                  # 客群列表
        data_list=data_list,                  # 数据源列表
        feature_list=feature_list,            # 融合模型特征前缀
        file_data_list=file_data_list,        # 数据文件命名
        file_base_name=file_base_name,        # 文件基础名
        ronghe_list=ronghe_list,              # 融合模型列表
        model_report_path=model_report_path,  # 模型报告路径
        data_or_path=data_or_path,            # 数据文件路径
        save_path=save_path,                  # 结果保存路径
        debug=debug                           # 调试模式开关
    )
    
    # 2. 执行完整打分流程（单模型 + 融合模型）
    all_score_result = scorer.get_score()
    
    # 3. 查看最终结果的所有列
    print("\n" + "=" * 60)
    print("打分完成！最终结果包含以下列:")
    print("=" * 60)
    print(all_score_result.columns.tolist())
    print("=" * 60)

    # ==================== Jupyter Notebook 便捷复制代码 ====================
    # 以下代码可直接复制到 Jupyter Notebook 中使用（已注释，取消注释即可运行）
    # 
    # import pandas as pd
    # from new_tools.score import TzScore  # 确保 score.py 在当前工作目录或 Python 路径中
    # 
    # debug = True
    # ex_lst = ['mobile', 'backPointTime']  # 关键列：用于去重和连接
    # cust_list = ['api','app','xxl','all']  # 客群列表
    # data_list = ['deltaV1','thetaV2','zetaV2']  # 数据源列表（对应模型文件夹）
    # feature_list = ['delta','theta','zeta']  # 融合模型特征前缀
    # file_data_list = ['deltaV1','thetaV2','zetaV2']  # 数据文件命名
    # ronghe_list = ['ronghe']  # 融合模型列表
    # file_base_name = 'rz'  # 数据文件基础名
    # model_report_path = r'D:\work\06.std_product\xyf\all results\xyf_better/'
    # data_or_path = r'D:\work\06.std_product\918_std_product\rz/'
    # save_path = r'D:\work\06.std_product\xyf\tmp_score_test/'
    # 
    # # 加载标签数据并预处理
    # label = pd.read_csv(data_or_path + r'\rz_50w_20240501_20241230_fpd30_dpd30.csv')
    # label['app_dt'] = pd.to_datetime(label['app_dt'].astype('int').astype('str'))
    # label.rename(columns={'mobile_md5':'mobile','app_dt':'backPointTime'}, inplace=True)
    # 
    # # 执行打分
    # scorer = TzScore(label, ex_lst, cust_list, data_list, feature_list, file_data_list, 
    #                  file_base_name, ronghe_list, model_report_path, data_or_path, save_path, debug=debug)
    # all_score_result = scorer.get_score()
    # print(all_score_result.columns)
    # print(f"打分结果数据形状: {all_score_result.shape}")
